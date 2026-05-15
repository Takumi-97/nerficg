import os
from PIL import Image

image_dir = "."
files = os.listdir(image_dir)
count = 0

print("--- 画像ファイルを再エンコード中 ---")
for file in files:
    # .png ファイルのみを処理し、フォルダやその他のファイルを無視
    if file.endswith('.png') and file != 'convert_images.py':
        try:
            # 画像を読み込み (自動的に形式をデコード)
            img = Image.open(os.path.join(image_dir, file))
            # 同じ名前で保存し直し (より互換性の高いPNG形式でエンコード)
            img.save(os.path.join(image_dir, file), 'PNG')
            count += 1
        except Exception as e:
            print(f"警告: ファイル {file} の処理中にエラーが発生しました: {e}")

print(f"--- {count} 個の画像ファイルが再エンコードされました。 ---")
