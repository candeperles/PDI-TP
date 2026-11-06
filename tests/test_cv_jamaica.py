"""Persistencia y lectura de CV sin dataset, TensorFlow ni PyTorch."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from pdi_tp import clasificacion, comparacion, cv_jamaica as cv, split


class CvJamaicaTest(unittest.TestCase):
    def setUp(self):
        temporal = tempfile.TemporaryDirectory()
        self.addCleanup(temporal.cleanup)
        raiz = Path(temporal.name)
        self.rutas = SimpleNamespace(repo=raiz, imagenes=raiz / "images", analisis=raiz / "analisis",
            resultados_equipo=raiz / "resultados", solo_lectura=False)
        filas = [dict(archivo=f"kingston_{g * 10 + v}_0_0_0.png", split="val" if g < 2 else "train",
            y_original=v, pendiente_revision=False) for g in range(10) for v in range(2)]
        filas += [dict(archivo=f"PV01_{origen + clase * 20}_100.png", split=conjunto,
            y_original=clase, pendiente_revision=False)
            for conjunto, origen in (("train", 100), ("val", 300)) for clase in range(2)]
        self.tabla = pd.DataFrame(filas)
        for conjunto in ("train", "val"):
            (raiz / f"{conjunto}.txt").write_text("\n".join(self.tabla.loc[self.tabla.split.eq(conjunto), "archivo"]))
        for parche in (patch.object(split, "SPLITS_DIR", raiz),
                       patch.object(clasificacion, "preparar_tabla", return_value=(self.tabla, None)),
                       patch.object(cv.resultados, "_commit", return_value="a" * 40)):
            parche.start()
            self.addCleanup(parche.stop)
        self.folds = split.folds_jamaica(self.rutas)
        self.original_referencia = cv.referencia
        parche = patch.object(cv, "referencia", side_effect=self.referencia_falsa)
        parche.start()
        self.addCleanup(parche.stop)

    @staticmethod
    def referencia_falsa(rutas, modelo):
        return dict(receta=dict(epochs=50) if modelo in cv.VERSIONES_YOLO else dict(epocas_cabeza=30, epocas_fine_tuning=20),
                    referencias={"origen": "config_de_prueba"},
                    umbral=0.25 if modelo in cv.VERSIONES_YOLO else 0.5)

    def params(self, modelo="resnet50"):
        return dict(self.referencia_falsa(self.rutas, modelo), entorno={"paquetes": "prueba"},
                    huella_datos="huella_de_prueba", semilla_folds=42)

    def guardar(self, fold, modelo="resnet50", **cambios):
        modo, params = cv.FINALISTAS[modelo], self.params(modelo)
        params.update(cambios)
        carpeta = cv.preparar_fold(self.rutas, modelo, modo, fold, params)
        detector = modelo in cv.VERSIONES_YOLO
        (carpeta / ("modelo.pt" if detector else "modelo.keras")).write_bytes(b"pesos ficticios de test")
        etapas = {k: dict(epocas=2, mejor_epoca=1) for k in (("deteccion",) if detector else ("cabeza", "fine_tuning"))}
        probabilidades = fold["evaluacion"].y_original.map({0: 0.1, 1: 0.9})
        if detector:
            probabilidades = np.zeros(len(probabilidades))
        cv.guardar_fold(self.rutas, carpeta, modelo, modo, fold, params, probabilidades,
                        dict(etapas=etapas, segundos=0.1), cajas=[] if detector else None)
        return carpeta, params

    def test_guarda_y_carga_solo_cinco_folds_integros_sin_escribir_resultados(self):
        for fold in self.folds:
            carpeta, params = self.guardar(fold)
            self.assertTrue(cv.fold_completo(self.rutas, "resnet50", "imagenet_finetuning", fold, params))
            config = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["commit"], "a" * 40)
            self.assertIn("huella_fold", config)
        self.rutas.solo_lectura = True
        pred, estado = cv.cargar_predicciones(self.rutas, self.folds)
        self.assertEqual(len(pred), 20)
        self.assertFalse(pred.archivo.duplicated().any())
        self.assertEqual(int(estado.completa.sum()), 5)
        self.assertFalse(self.rutas.resultados_equipo.exists())

    def test_parciales_no_entran_y_no_se_pisan(self):
        fold = self.folds[0]
        self.assertFalse(cv.fold_completo(self.rutas, "resnet50", "imagenet_finetuning", fold, self.params()))
        carpeta, _ = self.guardar(fold)
        pred, estado = cv.cargar_predicciones(self.rutas, self.folds)
        self.assertTrue(pred.empty)
        self.assertEqual(int(estado.completa.sum()), 1)
        with self.assertRaises(FileExistsError):
            cv.preparar_fold(self.rutas, "resnet50", "imagenet_finetuning", fold, self.params())
        anterior = (carpeta / "modelo.keras").read_bytes()
        nueva = cv.preparar_fold(self.rutas, "resnet50", "imagenet_finetuning", fold, self.params(), reiniciar=True)
        archivadas = list(carpeta.parent.glob("fold_1__anterior_*"))
        self.assertEqual(len(archivadas), 1)
        self.assertEqual((archivadas[0] / "modelo.keras").read_bytes(), anterior)
        self.assertFalse((nueva / "modelo.keras").exists())
        with self.assertRaises(FileExistsError):
            cv.fold_completo(self.rutas, "resnet50", "imagenet_finetuning", fold, self.params())

    def test_solo_lectura_no_crea_ni_modifica(self):
        self.rutas.solo_lectura = True
        with self.assertRaises(PermissionError):
            cv.preparar_fold(self.rutas, "resnet50", "imagenet_finetuning", self.folds[0], self.params())
        self.assertFalse(self.rutas.analisis.exists())
        with self.assertRaises(PermissionError):
            cv.guardar_fold(self.rutas, Path("cualquier"), "resnet50", "imagenet_finetuning", self.folds[0], {}, [], {})

    def test_rechaza_pesos_faltantes_probabilidades_y_epocas_invalidas(self):
        fold, params = self.folds[0], self.params()
        carpeta = cv.preparar_fold(self.rutas, "resnet50", "imagenet_finetuning", fold, params)
        prob = np.zeros(len(fold["evaluacion"]))
        with self.assertRaisesRegex(ValueError, "pesos"):
            cv.guardar_fold(self.rutas, carpeta, "resnet50", "imagenet_finetuning", fold, params, prob, {})
        (carpeta / "modelo.keras").write_bytes(b"pesos")
        for probabilidades in ([0.5], [np.nan] * len(prob), [2] * len(prob)):
            with self.assertRaisesRegex(ValueError, "puntaje"):
                cv.guardar_fold(self.rutas, carpeta, "resnet50", "imagenet_finetuning", fold, params, probabilidades, {})
        with self.assertRaisesRegex(ValueError, "épocas"):
            cv.guardar_fold(self.rutas, carpeta, "resnet50", "imagenet_finetuning", fold, params, prob, {})
        self.assertFalse((carpeta / "completo.json").exists())

    def test_detectores_exigen_cajas_y_admiten_cero_detecciones(self):
        carpeta, params = self.guardar(self.folds[0], "yolo11n")
        self.assertEqual(list(pd.read_csv(carpeta / "cajas_oof.csv")), cv.COLUMNAS_CAJAS)
        cv.preparar_fold(self.rutas, "yolo11n", "coco_completo", self.folds[0], params, reiniciar=True)
        (carpeta / "modelo.pt").write_bytes(b"pesos")
        with self.assertRaisesRegex(ValueError, "cajas fuera de fold"):
            cv.guardar_fold(self.rutas, carpeta, "yolo11n", "coco_completo", self.folds[0], params,
                [0.1] * len(self.folds[0]["evaluacion"]), dict(segundos=1, etapas={"deteccion": dict(epocas=2, mejor_epoca=1)}))

    def test_puntaje_detector_debe_coincidir_con_sus_cajas(self):
        fold, modelo, modo = self.folds[0], "yolo11n", "coco_completo"
        params = self.params(modelo)
        carpeta = cv.preparar_fold(self.rutas, modelo, modo, fold, params)
        (carpeta / "modelo.pt").write_bytes(b"pesos ficticios")
        probabilidades = np.zeros(len(fold["evaluacion"]))
        filas = [dict(archivo=fold["evaluacion"].archivo.iloc[0], clase=0, confianza=0.8,
                      xc=0.5, yc=0.5, w=0.2, h=0.2)]
        entrenamiento = dict(segundos=1, etapas={"deteccion": dict(epocas=2, mejor_epoca=1)})
        with self.assertRaisesRegex(ValueError, "máxima confianza"):
            cv.guardar_fold(self.rutas, carpeta, modelo, modo, fold, params, probabilidades, entrenamiento, cajas=filas)
        probabilidades[0] = 0.8
        cv.guardar_fold(self.rutas, carpeta, modelo, modo, fold, params, probabilidades, entrenamiento, cajas=filas)
        self.assertTrue(cv.fold_completo(self.rutas, modelo, modo, fold, params))

    def test_rechaza_cambios_de_receta_o_archivos_y_mezcla_de_commits(self):
        fold = self.folds[0]
        carpeta, params = self.guardar(fold)
        with self.assertRaisesRegex(ValueError, "receta de la referencia"):
            cv.fold_completo(self.rutas, "resnet50", "imagenet_finetuning", fold, dict(params, receta={"otra": 1}))
        with (carpeta / "predicciones_oof.csv").open("a") as entrada:
            entrada.write("dato adulterado\n")
        with self.assertRaisesRegex(ValueError, "modificado"):
            cv.cargar_predicciones(self.rutas, self.folds)
        cv.preparar_fold(self.rutas, "resnet50", "imagenet_finetuning", fold, params, reiniciar=True)
        # El resto se guarda bajo otro commit; no puede formar una corrida conjunta.
        for f in self.folds[1:]:
            self.guardar(f)
        with patch.object(cv.resultados, "_commit", return_value="b" * 40):
            cv.preparar_fold(self.rutas, "resnet50", "imagenet_finetuning", fold, params, reiniciar=True)
            # guardar() requiere un destino vacío; archivar esta solicitud primero.
            carpeta.rename(carpeta.with_name("solicitud_archivada"))
            self.guardar(fold)
        with self.assertRaisesRegex(ValueError, "distinta receta"):
            cv.cargar_predicciones(self.rutas, self.folds)

    def test_rechaza_particiones_con_fuga_test_o_pendientes(self):
        for cambio in ("fuga", "test", "pendiente"):
            fold = {k: v.copy() if isinstance(v, pd.DataFrame) else v for k, v in self.folds[0].items()}
            if cambio == "fuga":
                fold["train"] = pd.concat([fold["train"], fold["evaluacion"].iloc[[0]]])
            elif cambio == "test":
                fold["evaluacion"].loc[0, "archivo"] = "kingston_9999_0_0_0.png"
            else:
                fold["train"].loc[0, "pendiente_revision"] = True
            with self.subTest(cambio=cambio), self.assertRaises(ValueError):
                cv.preparar_fold(self.rutas, "resnet50", "imagenet_finetuning", fold, self.params())

    def test_tabla_umbral_compara_yolo_con_025(self):
        jamaica = self.tabla.loc[self.tabla.archivo.map(split.es_jamaica)]
        pred = jamaica.assign(modelo="yolo11n", modo="coco_completo", y_real=jamaica.y_original,
            probabilidad=jamaica.y_original.map({0: 0.3, 1: 0.9}), umbral=0.25)
        tabla = cv.tabla_umbrales(pred, n=30)
        self.assertEqual(tabla.iloc[0].umbral_referencia, 0.25)
        self.assertEqual(tabla.iloc[0].balanced_accuracy_referencia, 0.5)
        self.assertEqual(tabla.iloc[0].balanced_accuracy_05, 1)

    def test_intervalos_incluyen_recall_con_paneles(self):
        pred = self.tabla.assign(modelo="m", modo="a", y_real=self.tabla.y_original,
            probabilidad=self.tabla.y_original, prediccion=self.tabla.y_original)
        tabla = comparacion.intervalos(pred, ["m/a"], metricas=("recall",), n=30)
        self.assertEqual(tabla.iloc[0]["recall"], 1)
        self.assertIn("recall_inf", tabla)
        with self.assertRaises(ValueError):
            comparacion.intervalos(pred, ["m/a"], metricas=("inventada",))

    def test_receta_normaliza_resnet_y_mobilenet_sin_usar_sus_pesos(self):
        comunes = dict(receta="comun_v2", img_size=256, batch_size=32, dropout=0.3,
            paciencia=10, semilla=42, umbral=0.5)
        for modelo in ("resnet50", "mobilenetv2"):
            with self.subTest(modelo=modelo):
                params = dict(comunes, augmentation="aumentacion.google_maps",
                              pesos_muestra="igual_masa_por_fuente_x_clase")
                cabeza = dict(params, epocas_max=30, lr=1e-3)
                fine = dict(params, epocas_max=20, lr=1e-5)
                if modelo == "mobilenetv2":
                    params = dict(comunes, augmentation=True, sample_weight="igual_masa_por_fuente_x_clase",
                        epocas_cabeza=30, lr_cabeza=1e-3, epocas_fine_tuning=20, lr_fine_tuning=1e-5)
                    cabeza = fine = params
                for modo, p in (("imagenet_congelado", cabeza), ("imagenet_finetuning", fine)):
                    carpeta = self.rutas.resultados_equipo / modelo / modo
                    carpeta.mkdir(parents=True)
                    (carpeta / "config.json").write_text(json.dumps(dict(params=p)), encoding="utf-8")
                with patch.object(cv.resultados, "corrida_completa", return_value=True):
                    ref = self.original_referencia(self.rutas, modelo)
                    self.assertEqual(ref["receta"]["epocas_cabeza"], 30)
                    self.assertEqual(ref["receta"]["lr_fine_tuning"], 1e-5)
                    self.assertEqual(ref["receta"]["sample_weight"], "igual_masa_por_fuente_x_clase")
                    self.assertEqual(set(ref["referencias"]), {"imagenet_congelado", "imagenet_finetuning"})
                    fine = dict(fine, receta="vieja")
                    carpeta = self.rutas.resultados_equipo / modelo / "imagenet_finetuning"
                    (carpeta / "config.json").write_text(json.dumps(dict(params=fine)), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "comun_v2"):
                        self.original_referencia(self.rutas, modelo)

    def test_receta_detector_conserva_todos_los_parametros_y_exige_version(self):
        receta = dict(imgsz=640, epochs=50, patience=10, conf=0.001, iou=0.7, max_det=300,
            batch=16, optimizer="AdamW", lr0=0.001, hsv_s=0.4, mosaic=0)
        params = dict(receta=receta, freeze=0, umbral_clasificacion=0.25,
            entorno=dict(paquetes=dict(ultralytics="8.4.184")))
        carpeta = self.rutas.resultados_equipo / "yolo26n" / "coco_completo"
        carpeta.mkdir(parents=True)
        archivo = carpeta / "config.json"
        archivo.write_text(json.dumps(dict(params=params)), encoding="utf-8")
        with patch.object(cv.resultados, "corrida_completa", return_value=True):
            self.assertEqual(self.original_referencia(self.rutas, "yolo26n")["receta"], receta)
            params["entorno"]["paquetes"]["ultralytics"] = "8.3.221"
            archivo.write_text(json.dumps(dict(params=params)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Ultralytics"):
                self.original_referencia(self.rutas, "yolo26n")


if __name__ == "__main__":
    unittest.main()
