from .adni import build as build_adni


def build_dataset(image_set, args):
    if args.dataset_file == 'ADNI':
        return build_adni(image_set, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')


