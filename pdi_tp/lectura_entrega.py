"""Lectura de corridas para la entrega, sin TensorFlow, GPU ni escrituras."""
import hashlib
import json

import numpy as np
import pandas as pd

from . import entorno, resultados


def mostrar_modelo(modelo):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    rutas = entorno.preparar(solo_lectura=True)
    modos = (["coco_completo", "coco_backbone_congelado"] if modelo.startswith("yolo") else
             ["imagenet_congelado", "imagenet_finetuning", "aleatorio"])
    tabla = resultados.cargar_corridas(rutas)
    propias = tabla[tabla.modelo.eq(modelo) & tabla.modo.isin(modos)]
    if len(propias) != len(modos) or not propias.completa.all():
        raise ValueError(f"Faltan corridas completas de {modelo}; no se entrenó ni se modificó nada.")
    from IPython.display import display
    print("Consulta de corridas guardadas. No se volvió a entrenar ni a predecir.")
    print("Val eligió checkpoints y se usa para reportar; Jamaica es un subconjunto de val.")
    display(propias.reindex(columns=["modelo", "modo", "fecha", "completa", "f1_macro", "balanced_accuracy",
        "recall", "recall_sin_paneles", "roc_auc", "jamaica.f1_macro"]))
    for modo in modos:
        carpeta = resultados.ruta_corrida(rutas, modelo, modo)
        config = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
        metricas = json.loads((carpeta / "metricas.json").read_text(encoding="utf-8"))
        print(f"\n{modelo}/{modo} · fecha original: {config.get('fecha', 'no registrada')} · commit: {config.get('commit', 'no registrado')}")
        print("SHA-256 de las fuentes consultadas:")
        for nombre in ["config.json", "metricas.json", "predicciones_val.csv"]:
            print(nombre, hashlib.sha256((carpeta / nombre).read_bytes()).hexdigest())
        display(pd.json_normalize(config))
        figura, eje = plt.subplots(figsize=(4.2, 3.5))
        ConfusionMatrixDisplay(np.array(metricas["matriz_confusion"]), display_labels=["sin paneles", "con paneles"]).plot(ax=eje, colorbar=False)
        eje.set_title(f"{modelo}/{modo}", fontsize=9)
        figura.tight_layout()
        plt.show()
        plt.close(figura)
        historia = carpeta / "historia.csv"
        if historia.is_file():
            h = pd.read_csv(historia)
            figura, ejes = plt.subplots(1, 2, figsize=(10, 3))
            for eje, metrica in zip(ejes, ["loss", "accuracy"]):
                for columna in [metrica, "val_" + metrica]:
                    if columna in h:
                        eje.plot(np.arange(1, len(h) + 1), h[columna], label=columna)
                eje.set(xlabel="Época", title=metrica)
                if eje.lines:
                    eje.legend()
            figura.tight_layout()
            plt.show()
            plt.close(figura)
        if modelo.startswith("yolo"):
            entrenamiento = carpeta / "entrenamiento" / "results.png"
            if entrenamiento.is_file():
                from IPython.display import Image
                display(Image(filename=str(entrenamiento)))
            display(pd.DataFrame(json.loads((carpeta / "metricas_deteccion.json").read_text(encoding="utf-8"))).T)
            curva = carpeta / "deteccion_val" / "BoxPR_curve.png"
            if not curva.is_file():
                curva = carpeta / "deteccion_val" / "PR_curve.png"
            if curva.is_file():
                from IPython.display import Image
                display(Image(filename=str(curva)))
            else:
                print("Curva PR no disponible en esta copia de la corrida.")
