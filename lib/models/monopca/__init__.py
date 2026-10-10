def build_monopca(cfg):
    if cfg['type'] == 'monopca':
        from .monopca import build
        return build(cfg)
    else:
        raise ValueError(f"Invalid model type: {cfg['type']}")
