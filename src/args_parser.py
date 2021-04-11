import argparse


def parse_args(is_federated):
    parser = argparse.ArgumentParser()
    parse_common(parser)

    if is_federated == True:
        parse_federated(parser)
    elif is_federated == False:
        parse_baseline(parser)
    else:
        print('Unspecified mode: federated or baseline?')

    args = parser.parse_args()
    return args

def parse_baseline(parser):
    parser.add_argument(
        '--epochs', type=int, default=100, help="Number of epochs of training."
    )
    # parser.add_argument(
    #     '--batch_size', default=None, help="Number of items for each batch. If unspecified no batch is used."
    # )



def parse_federated(parser):
    parser.add_argument(
        '--rounds', type=int, default=10, help="Number of rounds of training."
    )
    parser.add_argument(
        '--K', type=int, default=100, help="Total number of clients."
    )
    parser.add_argument(
        '--C', type=float, default=0.1, help='Fraction of clients to employ in each round. From 0 to 1.'
    )

def parse_common(parser):
    parser.add_argument(
        '--dataset', type=str, default='iris', help="Name of the dataset."
    )
    parser.add_argument(
        '--components', type=int, default=3, help="Number of Gaussians to fit."
    )
    parser.add_argument(
        '--random_seed', default=None, help="Number to have random consistent results across executions."
    )
    # parser.add_argument(
    #     '--path', default=None, help="Path of the custom dataset to use."
    # )
    parser.add_argument(
        '--samples', type=int, default=10000, help="Number of samples to generate."
    )
    parser.add_argument(
        '--features', type=int, default=2, help="Number of features for each generated sample."
    )
    parser.add_argument(
        '--soft_clustering', type=int, default=1, help="Specifies if cluster bounds are soft or hard."
    )
    parser.add_argument(
        '--plots_3d', type=int, default=0, help="Specifies if plots are to be done in 3D or 2D."
    )
