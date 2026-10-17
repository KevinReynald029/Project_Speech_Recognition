import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phoneme_scoring.backend import create_backend
from phoneme_scoring.data import dataset_directory


class PortabilityTests(unittest.TestCase):
    def test_shared_config_paths_do_not_depend_on_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'config').mkdir()
            file = root/'config/native.json'
            file.write_text(json.dumps(dict(backend='native',project_root='..',bin='runtime/bin',model='runtime/model',lang='runtime/lang',ivector_extractor='runtime/extractor',scratch='.cache/scratch')))
            os.chdir(root/'config')
            try:
                with patch('phoneme_scoring.native.NativeKaldiBackend') as backend:
                    create_backend(file)
                    config=backend.call_args.args[0]
                    self.assertEqual(Path(config['bin']),root/'runtime/bin')
                    self.assertEqual(Path(config['scratch']),root/'.cache/scratch')
                    self.assertNotIn('project_root',config)
            finally:
                os.chdir(previous)

    def test_dictionary_config_is_not_mutated(self):
        config={'backend':'native','project_root':str(Path.cwd()),'bin':'runtime/bin'}
        original=dict(config)
        with patch('phoneme_scoring.native.NativeKaldiBackend'):
            create_backend(config)
        self.assertEqual(config,original)

    def test_dataset_metadata_can_be_loaded_from_repository_handoff(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            directory=root/'handoff'; directory.mkdir()
            for split in ['train','val','test']:
                (directory/f'manifest_{split}.jsonl').write_text('')
            self.assertEqual(dataset_directory(root),directory)
            local=root/'data/handoff'; local.mkdir(parents=True)
            for split in ['train','val','test']:
                (local/f'manifest_{split}.jsonl').write_text('')
            self.assertEqual(dataset_directory(root),local)

    def test_missing_handoff_has_clear_error(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError,'handoff'):
                dataset_directory(temp)
