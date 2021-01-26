import argparse
from deepgraphpose.models.eval import estimate_pose
from deepgraphpose.models.eval import plot_dgp


def run_main(args):

    if args.sample:
        plot_dgp(
            proj_cfg_file=args.proj_cfg_file, dgp_model_file=args.dgp_model_file,
            video_file=args.video_file, output_dir=args.output_dir, shuffle=int(args.shuffle))
    else:
        estimate_pose(
            proj_cfg_file=args.proj_cfg_file, dgp_model_file=args.dgp_model_file,
            video_file=args.video_file, output_dir=args.output_dir, shuffle=int(args.shuffle),
            save_pose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dgp project configuration file (.yaml)
    parser.add_argument('--proj_cfg_file', type=str)
    # dgp model file (no extension)
    parser.add_argument('--dgp_model_file', type=str)
    # video file to process
    parser.add_argument('--video_file', type=str)
    # directory to store labels/labeled video
    parser.add_argument('--output_dir', type=str)
    # dgp shuffle number
    parser.add_argument('--shuffle', default=2, type=str)
    # label sample video; if True, saves full labeled video
    parser.add_argument('--sample', action='store_true', default=False)
    namespace, _ = parser.parse_known_args()
    run_main(namespace)
