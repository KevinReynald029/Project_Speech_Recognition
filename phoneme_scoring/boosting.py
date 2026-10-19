from __future__ import annotations

from pathlib import Path
import numpy as np

from .common import FEATURE_DIM, FEATURE_VERSION, METHOD_VERSION, PHONES, PREPROCESS_VERSION, ScoringError, dump_json, load_json


def export_histogram_model(model, *, shrink_to_median=0., train_median=1.):
    """Nilai leaf dari scikit-learn sudah memuat learning rate."""
    trees = []
    for stage in model._predictors:
        if len(stage) != 1:
            raise ValueError("Only single-output regression is supported")
        nodes = stage[0].nodes
        if nodes['is_categorical'].any():
            raise ValueError("Categorical tree splits cannot be exported")
        trees.append([[int(n['feature_idx']), float(n['num_threshold']), int(n['left']), int(n['right']), float(n['value']), bool(n['is_leaf'])] for n in nodes])
    return dict(model_type='histogram_gradient_boosting', initial_prediction=float(model._baseline_prediction[0, 0]), trees=trees, shrink_to_median=float(shrink_to_median), train_median=float(train_median), tree_node_fields=['feature','threshold','left','right','value','is_leaf'])


class BoostedCalibrator:
    def __init__(self, artifact):
        try:
            a = load_json(artifact) if isinstance(artifact, (str, bytes, Path)) else artifact
            if a['method_version'] != METHOD_VERSION or a['model_type'] != 'histogram_gradient_boosting' or a['feature_version'] != FEATURE_VERSION or a['preprocess_version'] != PREPROCESS_VERSION or a['phone_order'] != list(PHONES):
                raise ValueError('Incompatible boosted calibrator')
            if not a.get('extractor_id') or not a.get('threshold_locked_on_validation'):
                raise ValueError('Acoustic identity and validation threshold are required')
            self.initial = float(a['initial_prediction'])
            self.shrink, self.median = float(a['shrink_to_median']), float(a['train_median'])
            self.threshold = float(a['practice_threshold_norm'])
            if not np.isfinite([self.initial,self.shrink,self.median,self.threshold]).all() or not 0 <= self.shrink <= 1 or not 0 <= self.median <= 1 or not 0 <= self.threshold <= np.nextafter(1.,np.inf):
                raise ValueError('Invalid boosted calibrator constants')
            self.trees = []
            if not 1 <= len(a['trees']) <= 2000:
                raise ValueError('Invalid number of trees')
            for tree in a['trees']:
                t = np.asarray(tree, dtype=float)
                if t.ndim != 2 or t.shape[1] != 6 or not 1 <= len(t) <= 4095 or not np.isfinite(t).all():
                    raise ValueError('Invalid tree nodes')
                if not np.isin(t[:,5],[0,1]).all():
                    raise ValueError('Invalid leaf flag')
                internal = np.flatnonzero(t[:,5] == 0)
                if np.any(t[internal,0] != np.floor(t[internal,0])) or np.any((t[internal,0]<0)|(t[internal,0]>=FEATURE_DIM)):
                    raise ValueError('Invalid tree feature index')
                for child in (2,3):
                    if np.any(t[internal,child] != np.floor(t[internal,child])) or np.any((t[internal,child]<=internal)|(t[internal,child]>=len(t))):
                        raise ValueError('Invalid or cyclic tree edge')
                self.trees.append(t)
            self.extractor_id, self.artifact = a['extractor_id'], a
        except (KeyError,TypeError,ValueError,OSError,IndexError) as error:
            raise ScoringError('model_not_ready',str(error)) from error

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != FEATURE_DIM or not np.isfinite(x).all():
            raise ScoringError('alignment_failed','Invalid acoustic features')
        prediction = np.full(len(x), self.initial)
        for t in self.trees:
            indices = np.zeros(len(x), dtype=int)
            active = np.flatnonzero(t[indices,5] == 0)
            while len(active):
                nodes = t[indices[active]]
                left = x[active,nodes[:,0].astype(int)] <= nodes[:,1]
                indices[active] = np.where(left,nodes[:,2],nodes[:,3]).astype(int)
                active = active[t[indices[active],5] == 0]
            prediction += t[indices,4]
        prediction = (1-self.shrink)*np.clip(prediction,0,1)+self.shrink*self.median
        if not np.isfinite(prediction).all():
            raise ScoringError('model_not_ready','Nonfinite boosted prediction')
        return prediction

    def save(self, path):
        dump_json(path,self.artifact)


def load_calibrator(artifact):
    try:
        a = load_json(artifact) if isinstance(artifact,(str,bytes,Path)) else artifact
    except (OSError,ValueError,TypeError) as error:
        raise ScoringError('model_not_ready',str(error)) from error
    if not isinstance(a,dict):
        raise ScoringError('model_not_ready','Calibrator JSON must be an object')
    if a.get('model_type') == 'histogram_gradient_boosting':
        return BoostedCalibrator(a)
    from .model import RidgeCalibrator
    return RidgeCalibrator(a)


def fit_candidates(x_train, y_train, x_val, y_val, metadata):
    """24 kandidat, dipilih lewat MAE validation."""
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    candidates, best = [], None
    median = float(np.median(y_train))
    with threadpool_limits(limits=1):
        for loss in ('absolute_error','squared_error'):
            for leaves, minimum in ((7,100),(15,50),(31,30)):
                params = dict(loss=loss,max_iter=250,max_leaf_nodes=leaves,min_samples_leaf=minimum,learning_rate=.08,l2_regularization=5,early_stopping=False,random_state=2026)
                model = HistGradientBoostingRegressor(**params).fit(x_train,y_train)
                base = np.clip(model.predict(x_val),0,1)
                for shrink in (0.,.25,.5,.75):
                    prediction = (1-shrink)*base+shrink*median
                    mae = float(np.abs(prediction-y_val).mean())
                    candidate = dict(parameters=params,shrink_to_median=shrink,validation_mae_raw_0_2=2*mae)
                    candidates.append(candidate)
                    if best is None or mae < best[0]-1e-12:
                        artifact = {**metadata,**export_histogram_model(model,shrink_to_median=shrink,train_median=median), 'fit_parameters':params,'training_sklearn_version':sklearn.__version__}
                        portable = BoostedCalibrator(artifact).predict(x_val)
                        if not np.allclose(prediction,portable,rtol=0,atol=1e-12):
                            raise ValueError('Portable tree predictions differ from sklearn')
                        best = (mae,artifact,prediction.copy())
                print(f"Validation: loss={loss}, leaves={leaves}, best MAE={best[0]*2:.6f}",flush=True)
    best[1]['candidate_validation_results'] = candidates
    return best[1],best[2]
