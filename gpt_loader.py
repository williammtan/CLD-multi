import numpy as np
def prepare_10class(X_train, y_train, X_test, y_test, class_ids=None):
    """
    X_*: images or features; y_*: original labels (ints or strings)
    class_ids: list/array of the 10 class labels to keep (in desired order)
               If None, we infer 10 unique labels from y_train.
    Returns flattened, float32-scaled X and integer y in [0..9].
    """
    # 1) choose 10 classes
    if class_ids is None:
        uniq = np.unique(y_train)
        assert len(uniq) >= 10, "Need at least 10 distinct classes"
        class_ids = uniq[:10]
    class_ids = list(class_ids)
    # 2) filter to those classes
    def _filter(X, y):
        mask = np.isin(y, class_ids)
        return X[mask], y[mask]
    X_train, y_train = _filter(X_train, y_train)
    X_test,  y_test  = _filter(X_test,  y_test)
    # 3) remap labels -> 0..9 (in the order of class_ids)
    remap = {c:i for i, c in enumerate(class_ids)}
    y_train = np.array([remap[c] for c in y_train], dtype=np.int32)
    y_test  = np.array([remap[c] for c in y_test],  dtype=np.int32)
    # 4) flatten + scale if these are images
    X_train = X_train.reshape(len(X_train), -1).astype(np.float32)
    X_test  = X_test.reshape(len(X_test),  -1).astype(np.float32)
    if X_train.max() > 1.0:  # simple normalization for raw pixels
        X_train /= 255.0
        X_test  /= 255.0
    return X_train, y_train, X_test, y_test