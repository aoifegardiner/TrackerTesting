import argparse
from SurgT.src.utils import load_yaml_data, download_data
from SurgT.src.evaluate import evaluate_method


def main():
    parser = argparse.ArgumentParser(description='Tool to benchmark soft tissue trackers')
    parser.add_argument('--config', type=str, default='SurgT/config.yaml')
    parser.add_argument('-nv', '--no-visualization', help="no visualization is shown.", action="store_true")
    args = parser.parse_args()
    config = load_yaml_data(args.config)
    # Download data (if not downloaded before)
    download_data(config)
    # Run method
    is_visualization_off = True
    evaluate_method(config, is_visualization_off)


if __name__ == "__main__":
    main()
