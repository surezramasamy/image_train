# YOLO Studio
Run `run.bat` (Windows) or `run.sh`. Opens http://127.0.0.1:8000.

## One-time Google setup (needed for "Sign in with Google")
1. https://console.cloud.google.com -> new project -> enable **Google Drive API**.
2. OAuth consent screen: External, add your Gmail as a Test user.
3. Credentials -> Create OAuth client ID -> **Desktop app** -> download JSON -> save as `client_secret.json` in this folder.
4. In the app: Colab setup tab -> Sign in with Google.

## Workflow
Sign in -> create project -> upload images -> annotate (auto-synced to Drive/YoloStudio/projects) ->
"Open in Colab" (notebook opens straight from your Drive; set GPU, Run all) -> Train tab -> start.
Weights (best.pt, ONNX) and precision/recall/mAP are downloaded automatically.
