import argparse

from signdiff import  CVT_test



def main():
    # Example options:
    # train ./Configs/Base.yaml
    # test ./Configs/Base.yaml

    ap = argparse.ArgumentParser("Progressive Transformers")

    # Choose between Train and Test
    ap.add_argument("mode", choices=["train", "test", "pre_train","pre_test", "CVT", "CVT_test"],
                    help="train a model or test")
    # Path to Config
    ap.add_argument("config_path", type=str,
                    help="path to YAML config file")

    # Optional path to checkpoint
    ap.add_argument("--ckpt", type=str,
                    help="path to model checkpoint")

    ap.add_argument("--talent_ckpt", type=str,
                    help="path to pre model checkpoint")

    args = ap.parse_args()

    # If Train
    if args.mode == "CVT_test":
        CVT_test(cfg_file=args.config_path, ckpt=args.ckpt)
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
