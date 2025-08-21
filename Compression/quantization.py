import argparse
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize a model using GPTQ with customizable parameters."
    )

    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Number of bits for quantization (e.g., 2, 4, 8).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    model_id = "/home/fit/renju/WORK/lxm/models/DeepSeek_V2_Lite"
    # quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/fit/renju/WORK/lxm/models/DeepSeek_V2_Lite"
    )
    calibration_dataset = load_dataset(
        "allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split="train"
    ).select(range(1024))["text"]

    # dynamic = {
    #     r"+:.*\.1\..*attn.*": {"bits": 4, "group_size": 128},
    #     # r".*\.19\..*gate.*": {"bits": 8, "group_size": 64},
    #     r"-:.*self_attn.*": {},
    #     # r"-:.*down.*": {},
    # }
    dynamic = {
        # 只对 layer1 量化
        # r"+:.*\.1\.*attn.*": {"bits": 4, "group_size": 128},
        # # 跳过所有其他层
        # r"-:.*\.[0-9]\..*": {},          # 匹配 0, 2~9 层
        # r"-:.*\.1[0-9]\..*": {},          # 匹配 10~19
        # r"-:.*\.2[0-9]\..*": {},          # 匹配 20~29
        # r"-:.*\.3[0-9]\..*": {},          # 匹配 30~39 等
    }

    bits = args.bits
    quant_config = QuantizeConfig(bits=bits, group_size=128, dynamic=dynamic)
    print(quant_config)
    print(dynamic)
    model = GPTQModel.load(model_id, quant_config, trust_remote_code=True)

    # increase `batch_size` to match gpu/vram specs to speed up quantization
    model.quantize(calibration_dataset, batch_size=1)

    # model.save(quant_path)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    model.save(
        f"/home/fit/renju/WORK/lxm/Compression/models/DeepSeek_V2_Lite-all{bits}bit"
    )


if __name__ == "__main__":
    main()
