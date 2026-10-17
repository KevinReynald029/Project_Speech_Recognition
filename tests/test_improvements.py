import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from phoneme_scoring.boosting import BoostedCalibrator, export_histogram_model, load_calibrator
from phoneme_scoring.common import FEATURE_DIM, FEATURE_VERSION, METHOD_VERSION, PHONES, PREPROCESS_VERSION, ScoringError
from phoneme_scoring.native import NativeKaldiBackend


def metadata():
    return dict(method_version=METHOD_VERSION, feature_version=FEATURE_VERSION, preprocess_version=PREPROCESS_VERSION, phone_order=list(PHONES), extractor_id='synthetic-test-only', practice_threshold_norm=.4, threshold_locked_on_validation=True)


class PortableBoostingTests(unittest.TestCase):
    def fitted(self):
        rng = np.random.default_rng(43)
        x = rng.normal(size=(240,FEATURE_DIM))
        y = np.where(x[:,0] > .2,1.,np.clip(.6+.1*x[:,1],0,1))
        with threadpool_limits(limits=1):
            model = HistGradientBoostingRegressor(loss='absolute_error',max_iter=15,max_leaf_nodes=7,min_samples_leaf=8,early_stopping=False,random_state=43).fit(x,y)
        return model,x

    def test_json_roundtrip_matches_sklearn_on_unseen_and_split_boundaries(self):
        model,_ = self.fitted()
        x = np.random.default_rng(44).normal(size=(60,FEATURE_DIM))
        for stage in model._predictors:
            for node in stage[0].nodes:
                if not node['is_leaf']:
                    x[0,int(node['feature_idx'])] = node['num_threshold']
                    break
        a = {**metadata(),**export_histogram_model(model,shrink_to_median=.25,train_median=1.)}
        expected = .75*np.clip(model.predict(x),0,1)+.25
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'model.json'
            BoostedCalibrator(a).save(path)
            np.testing.assert_allclose(expected,load_calibrator(path).predict(x),rtol=0,atol=1e-12)

    def test_rejects_cycles_nonfinite_and_wrong_identity(self):
        model,_ = self.fitted()
        for invalid in [None, [], 3.14]:
            with self.assertRaises(ScoringError): load_calibrator(invalid)
        original = {**metadata(),**export_histogram_model(model)}
        mutations = [lambda a:a['trees'][0][0].__setitem__(2,0), lambda a:a['trees'][0][0].__setitem__(1,float('nan')),lambda a:a.update(extractor_id=''),lambda a:a.update(threshold_locked_on_validation=False)]
        for mutate in mutations:
            a = copy.deepcopy(original); mutate(a)
            with self.assertRaises(ScoringError): BoostedCalibrator(a)


class RetryAlignmentTests(unittest.TestCase):
    def backend(self, outputs):
        b = object.__new__(NativeKaldiBackend)
        b.paths = {'model':Path('model')}
        self.calls = []
        def run(name,args,root):
            self.calls.append((name,args))
            if name == 'align-compiled-mapped':
                attempt = 'wide' if '--beam=100' in args else 'initial'
                if outputs.get(attempt) == 'timeout':
                    raise ScoringError('backend_timeout','timeout')
                if outputs.get(attempt):
                    Path(str(args[-1])[4:]).write_bytes(b'fixture')
                else:
                    raise ScoringError('alignment_failed','no final path')
            else:
                attempt = 'wide' if 'ali_wide' in str(args[-2]) else 'initial'
                Path(str(args[-1])[6:]).write_text(outputs[attempt])
        b.run = run
        return b

    def test_partial_retry_keeps_original_successful_path(self):
        b = self.backend({'initial':'a 12 3\n','wide':'a 12 9\nb 13 4\n'})
        with tempfile.TemporaryDirectory() as temp:
            spans,recovered = b.align_with_retry(Path(temp),str,['a','b'])
        self.assertEqual(spans,{'a':[(12,3)],'b':[(13,4)]})
        self.assertEqual(recovered,{'b'})

    def test_total_initial_failure_can_recover(self):
        b = self.backend({'initial':'','wide':'a 12 4\n'})
        with tempfile.TemporaryDirectory() as temp:
            spans,recovered = b.align_with_retry(Path(temp),str,['a'])
        self.assertEqual(recovered,{'a'})
        self.assertEqual(spans['a'],[(12,4)])

    def test_success_does_not_retry_and_unrecoverable_has_no_fake_path(self):
        for outputs,expected_calls in [({'initial':'a 12 3\n'},1),({'initial':'','wide':''},2)]:
            b=self.backend(outputs)
            with tempfile.TemporaryDirectory() as temp:
                spans,recovered=b.align_with_retry(Path(temp),str,['a'])
            self.assertEqual(sum(n=='align-compiled-mapped' for n,_ in self.calls),expected_calls)
            if expected_calls==2: self.assertEqual(spans,{})
            self.assertEqual(recovered,set())

    def test_timeout_is_not_misreported_as_pronunciation(self):
        b=self.backend({'initial':'timeout'})
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ScoringError) as ctx:
            b.align_with_retry(Path(temp),str,['a'])
        self.assertEqual(ctx.exception.code,'backend_timeout')
