import io

import torch
from aiohttp import web
from PIL import Image
from torchvision import models, transforms

# GPUの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 事前学習済みモデルをロード
model = models.resnet50(pretrained=True).to(device)
model.eval()

# 入力画像の前処理
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ImageNetクラスラベルのロード
imagenet_classes = {
    idx: name
    for idx, name in enumerate(open("imagenet_classes.txt").read().splitlines())
}


async def handle_predict(request):
    """画像を受け取り、モデルで推論を行うハンドラー"""
    try:
        # アップロードされたファイルを取得
        reader = await request.multipart()
        file = await reader.next()
        if not file or file.name != "file":
            return web.json_response(
                {"error": "File upload with key 'file' is required"}, status=400
            )

        # ファイルデータの読み込み
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")

        # 前処理
        input_tensor = transform(image).unsqueeze(0).to(device)

        # 推論
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted_idx = outputs.max(1)
            predicted_class = imagenet_classes[predicted_idx.item()]

        # 使用されたデバイスを返す
        device_used = "GPU" if torch.cuda.is_available() else "CPU"

        return web.json_response({"class": predicted_class, "device_used": device_used})

    except Exception as e:
        return web.json_response(
            {"error": f"Error processing the image: {e}"}, status=500
        )


async def handle_health(request):
    return web.json_response({"status": "ok"})


app = web.Application()
app.router.add_get("/health", handle_health)
app.router.add_post("/predict", handle_predict)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=8000)
