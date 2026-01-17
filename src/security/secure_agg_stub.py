class SecureAggregator:
    def __init__(self):
        pass

    def aggregate(self, state_dicts):
        if not state_dicts:
            return None
        base = {k: v.clone() for k, v in state_dicts[0].items()}
        for k in base.keys():
            for sd in state_dicts[1:]:
                base[k] += sd[k]
            base[k] = base[k] / float(len(state_dicts))
        return base
