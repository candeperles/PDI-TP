"""Separación espacial y ausencia de test/checkpoints en los folds adicionales."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from pdi_tp import clasificacion, split


class FoldsJamaicaTest(unittest.TestCase):
    def setUp(self):
        temporal = tempfile.TemporaryDirectory()
        self.addCleanup(temporal.cleanup)
        self.raiz = Path(temporal.name)
        self.rutas = SimpleNamespace(imagenes=self.raiz / "images")
        filas = []
        # Diez grupos separados con dos clases y dos vecinos por grupo.
        for grupo in range(10):
            for vecino in range(2):
                filas.append(dict(archivo=f"kingston_{grupo * 10 + vecino}_0_0_0.png",
                    split="val" if grupo < 2 else "train", y_original=vecino,
                    pendiente_revision=False))
        for conjunto, origen in (("train", 100), ("val", 300)):
            for clase in range(2):
                filas.append(dict(archivo=f"PV01_{origen + clase * 20}_100.png",
                    split=conjunto, y_original=clase, pendiente_revision=False))
        self.tabla = pd.DataFrame(filas)
        self.guardar_split()
        self.parche_split = patch.object(split, "SPLITS_DIR", self.raiz)
        self.parche_split.start()
        self.addCleanup(self.parche_split.stop)
        parche = patch.object(clasificacion, "preparar_tabla", side_effect=lambda _: (self.tabla.copy(), None))
        parche.start()
        self.addCleanup(parche.stop)

    def guardar_split(self):
        for conjunto in ("train", "val"):
            nombres = self.tabla.loc[self.tabla.split.eq(conjunto), "archivo"]
            (self.raiz / f"{conjunto}.txt").write_text("\n".join(nombres), encoding="utf-8")

    def test_cada_imagen_se_evalua_una_vez_y_los_grupos_no_se_cruzan(self):
        antes = {s: (self.raiz / f"{s}.txt").read_bytes() for s in ("train", "val")}
        folds = split.folds_jamaica(self.rutas)
        evaluadas = []
        for fold in folds:
            train, validacion, evaluacion = [fold[k] for k in ("train", "validacion", "evaluacion")]
            evaluadas.extend(evaluacion.archivo)
            self.assertFalse(set(train.grupo) & set(evaluacion.grupo))
            self.assertFalse(set(validacion.grupo) & (set(train.grupo) | set(evaluacion.grupo)))
            self.assertFalse(validacion.archivo.map(split.es_jamaica).any())
            jamaica = self.tabla.loc[self.tabla.archivo.map(split.es_jamaica), "archivo"]
            self.assertEqual(set(train.loc[train.archivo.map(split.es_jamaica), "archivo"]),
                             set(jamaica) - set(evaluacion.archivo))
            self.assertEqual(set(train.loc[~train.archivo.map(split.es_jamaica), "archivo"]),
                             set(self.tabla.query("split == 'train'").loc[
                                 lambda t: ~t.archivo.map(split.es_jamaica), "archivo"]))
            self.assertEqual(evaluacion.y_original.mean(), 0.5)
        self.assertEqual(len(evaluadas), len(set(evaluadas)))
        self.assertEqual(set(evaluadas), set(self.tabla.loc[self.tabla.archivo.map(split.es_jamaica), "archivo"]))
        for s, contenido in antes.items():
            self.assertEqual((self.raiz / f"{s}.txt").read_bytes(), contenido)

    def test_reproducible_aun_si_cambia_el_orden_del_manifiesto(self):
        primero = split.folds_jamaica(self.rutas)
        self.tabla = self.tabla.sample(frac=1, random_state=8)
        segundo = split.folds_jamaica(self.rutas)
        for a, b in zip(primero, segundo):
            for conjunto in ("train", "validacion", "evaluacion"):
                pd.testing.assert_frame_equal(a[conjunto], b[conjunto])

    def test_estratifica_grupos_con_una_sola_clase_sin_perder_su_correspondencia(self):
        jamaica = self.tabla.archivo.map(split.es_jamaica)
        self.tabla.loc[jamaica, "y_original"] = self.tabla.loc[jamaica, "archivo"].map(
            lambda nombre: (int(nombre.split("_")[1]) // 10) % 2)
        # Cinco grupos negativos y cinco positivos del mismo tamaño permiten equilibrio exacto.
        for fold in split.folds_jamaica(self.rutas):
            self.assertEqual(fold["evaluacion"].y_original.mean(), 0.5)

    def test_test_o_splits_inconsistentes_no_entran(self):
        nueva = self.tabla.iloc[[0]].assign(archivo="kingston_999_0_0_0.png", split="test")
        self.tabla = pd.concat([self.tabla, nueva], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "no admite test"):
            split.folds_jamaica(self.rutas)

    def test_rechaza_grupos_que_cruzan_train_y_validacion(self):
        self.tabla.loc[self.tabla.archivo.eq("PV01_300_100.png"), "archivo"] = "PV01_101_100.png"
        self.guardar_split()
        with self.assertRaisesRegex(ValueError, "grupos espaciales compartidos"):
            split.folds_jamaica(self.rutas)

    def test_rechaza_pendientes_jamaicanos_y_pocos_grupos(self):
        self.tabla.loc[0, "pendiente_revision"] = True
        with self.assertRaisesRegex(ValueError, "pendientes"):
            split.folds_jamaica(self.rutas)
        self.tabla.loc[0, "pendiente_revision"] = False
        for n in (0, 1, True, 1.5, 11):
            with self.subTest(n=n), self.assertRaises(ValueError):
                split.folds_jamaica(self.rutas, n=n)


if __name__ == "__main__":
    unittest.main()
