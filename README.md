# Reconnaissance de panneaux de limitation de vitesse

Projet de Vision par Ordinateur — détection automatique des panneaux **50, 70, 90 et 110 km/h** sur vidéos ou flux webcam en direct.

## Méthode

J'ai utilisé YOLOv8 (Ultralytics) pour la détection. Comme le dataset Roboflow fourni était au format classification (sans bounding boxes), j'ai fait :

1. **Pseudo-labelling** automatique avec un modèle YOLOv8 pré-entraîné sur panneaux européens (JakobJFL, HuggingFace), pour générer des annotations YOLO `.txt`.
2. **Fine-tuning** d'un YOLOv8 sur 1832 images annotées, 50 époques sur GPU NVIDIA RTX 4070.
3. **Inférence** côté Java avec ONNX Runtime + OpenCV (Bytedeco). Pas besoin de Python à l'exécution.
4. **Tracker FSM** (optionnel) pour valider une séquence attendue de panneaux (utile sur les vidéos cibles).

## Résultats

| Validation (pendant l'entraînement) | Valeur |
|---|---|
| mAP@0.5 | **0.955** |
| mAP@0.5:0.95 | 0.842 |
| Précision | 0.947 |
| Rappel | 0.956 |

Vitesse d'inférence : **~16 FPS sur CPU**, ~50 FPS sur GPU.

## Stack

- **Java 17** + **Maven**
- **OpenCV-Java 4.9** (Bytedeco) pour la lecture vidéo, webcam et dessin des bbox
- **ONNX Runtime 1.17** pour l'inférence
- **PyTorch + Ultralytics** côté Python uniquement pour l'entraînement
- **picocli** pour la CLI

## Lancer le programme

### 1. Compiler

```powershell
mvn clean package
```

Crée `target/vision-ensem.jar` (fat-jar ~940 MB avec les natifs OpenCV/FFmpeg/ONNX).

### 2. Mode démo — jeu 3D « Driving Sim »

```powershell
java -jar target/vision-ensem.jar game
```

Une voiture Twizy roule sur un circuit aléatoire avec des panneaux 50/70/90/110. Le modèle YOLO détecte les panneaux en direct sur le rendu 3D, dessine les bbox, et la limite affichée se met à jour. **TAB** : pilotage automatique (la voiture suit la route et adapte sa vitesse aux panneaux). **R** : nouveau circuit. **ESC** : quitter.

### 3. Mode démo — webcam live

```powershell
java -jar target/vision-ensem.jar webcam
```

Ouvre une fenêtre qui affiche le flux de la webcam avec les détections superposées (bbox + label + confiance + HUD FPS).
**Quitter** : appuyer sur **ESC** ou fermer la fenêtre.

Options utiles :

```powershell
# Webcam externe USB (index différent)
java -jar target/vision-ensem.jar webcam --device 1

# Résolution custom
java -jar target/vision-ensem.jar webcam --width 1280 --height 720

# Avec tracker séquentiel (affiche WAIT_X → DONE)
java -jar target/vision-ensem.jar webcam --track 50,70,90,110

# Seuil de confiance plus strict
java -jar target/vision-ensem.jar webcam --conf 0.50
```

### 4. Inférence sur une vidéo (mode générique, sans tracker)

```powershell
java -jar target/vision-ensem.jar infer-video --input videos/input/video1.wm
```

Produit `videos/output/video1_yolo.mp4` (vidéo annotée avec bbox + label) et `reports/video1_yolo.json`.

Avec tracker séquentiel (utile pour validation stricte d'un ordre attendu) :

```powershell
# video1 : doit voir 90 puis 70 puis 50
java -jar target/vision-ensem.jar infer-video --input videos/input/video1.wm --track 90,70,50

# video2 : doit voir 110
java -jar target/vision-ensem.jar infer-video --input videos/input/video2.wm --track 110
```

Options :

```powershell
# Sortie custom
java -jar target/vision-ensem.jar infer-video --input video.mp4 --output annotee.mp4

# Smoke test rapide (50 frames)
java -jar target/vision-ensem.jar infer-video --input video.mp4 --max-frames 50

# Seuil de confiance
java -jar target/vision-ensem.jar infer-video --input video.mp4 --conf 0.40

# CPU uniquement
java -jar target/vision-ensem.jar infer-video --input video.mp4 --no-cuda
```

### 5. Sanity check (extraction première frame)

```powershell
java -jar target/vision-ensem.jar smoke --input videos/input/video1.wm
```

Écrit la première frame en JPG dans `logs/`.

### 6. Tests unitaires

```powershell
mvn test
```

14 tests JUnit (CSV reader, NMS, postprocessor YOLO, tracker FSM, checkpoint, ETA).

## Sous-commandes disponibles

| Commande | Description |
|---|---|
| `game` | Jeu 3D « Driving Sim » avec Twizy + détection YOLO live + pilotage auto |
| `webcam` | Détection live depuis la webcam (fenêtre OpenCV) |
| `infer-video` | Inférence sur une vidéo (mode détection ou tracker FSM via `--track`) |
| `smoke` | Lecture vidéo + extraction première frame (vérification) |
| `prepare-data` | Pseudo-labelling du dataset Roboflow vers format YOLO |
| `eval` | Métriques mAP/P/R/F1 sur un split YOLO annoté |
| `compare` | Comparer plusieurs méthodes (YOLO vs Haar+CNN) |
| `bench` | Benchmark FPS / latence sur un dossier d'images |

## Organisation du code

```
src/main/java/fr/ensem/vision/
├── App.java                    point d'entrée CLI (picocli)
├── cli/                        arguments
├── config/                     ClassMapping, AppConfig
├── data/                       lecture CSV, pseudo-labelling
├── detection/                  Detection, BoundingBox
├── inference/yolo/             YOLO + ONNX Runtime
├── inference/haarcnn/          alternative Haar+CNN
├── video/                      lecture/écriture vidéo, webcam, annotation
├── tracker/                    FSM séquentielle (optionnelle)
├── eval/                       mAP, P/R/F1
└── util/                       checkpoint, ETA, logger, GC adaptatif
```

## Scripts Python (entraînement uniquement)

- `scripts/python/pseudo_label_dataset.py` : génère les labels YOLO à partir du CSV Roboflow + modèle pré-entraîné
- `scripts/python/finetune_from_jakob.py` : fine-tuning YOLOv8 (50 epochs)
- `scripts/python/test_on_photos.py` : évaluation sur un split avec métriques par classe

## Modèle fourni

- `models/yolov8_finetuned.onnx` (44.7 MB) — modèle final, 4 classes (50/70/90/110)
- `models/yolov8_finetuned.pt` (22.5 MB) — version PyTorch (au cas où)

## Limites connues

- Le modèle ne connaît que 4 classes (50/70/90/110). S'il voit un panneau 30 ou 100, il le classe vers la classe la plus proche (souvent 50 ou 90) → faux positifs possibles sur les images contenant d'autres limitations.
- Rappel par-image variable (~67% sur le test split). Le tracker FSM temporel (5 détections cohérentes sur 15 frames) compense largement quand on l'active.
- Les "crops" GTSRB du dataset ne sont pas détectables (YOLO n'est pas un classifieur de crops).

## Données sources

Dataset Roboflow Tazi (14 895 images, format classification). Modèle pré-entraîné panneaux européens : JakobJFL/yolov8-dk-Traffic-Signs (HuggingFace, Apache 2.0).
