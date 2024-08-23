
import argparse
import cv2
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--model_name', type=str, default='RealESRGAN_x4plus', help='Model name')
    parser.add_argument('--denoise_strength', type=float, default=0.7, help='Denoise strength')
    parser.add_argument('--tile', type=int, default=512, help='Tile size for memory optimization')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument('--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')

    args = parser.parse_args()

    # Load the model
    model = RealESRGANer(
        scale=args.outscale,
        model_path=args.model_name,
        denoise_strength=args.denoise_strength,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half_precision=not args.fp32
    )

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Inference
    input_path = args.input
    output_path = args.output
    suffix = args.suffix

    if os.path.isfile(input_path):
        paths = [input_path]
    else:
        paths = sorted([os.path.join(input_path, f) for f in os.listdir(input_path)])

    for img_path in paths:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        output, _ = model.enhance(img, outscale=args.outscale)
        save_path = os.path.join(output_path, f"{os.path.splitext(os.path.basename(img_path))[0]}_{suffix}.png")
        cv2.imwrite(save_path, output)

if __name__ == '__main__':
    main()
