"""Offline tests: real temporary Chroma, deterministic local embeddings only."""
import ast
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import retriever
from ingest import load_saved_chunks


class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, float(len(text) % 11), 0.5] for text in texts]

    def embed_query(self, text):
        raise AssertionError('Tests must not perform retrieval embedding calls')


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.a = Document(page_content='a safety procedure', metadata={'source_file': 'a.pdf', 'page': 0})
        self.b = Document(page_content='another procedure', metadata={'source_file': 'b.pdf', 'page': 1})
        self.fuse = retriever.HybridRetriever.__new__(retriever.HybridRetriever).reciprocal_rank_fusion

    def test_repeated_build_is_idempotent_and_startup_does_not_ingest(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            retriever, 'OpenAIEmbeddings', return_value=LocalEmbeddings()
        ):
            first = retriever.build_index([self.a, self.b, self.a], directory)
            ids = set(first.get()['ids'])
            second = retriever.build_index([self.b, self.a], directory)
            self.assertEqual(second._collection.count(), 2)
            self.assertEqual(set(second.get()['ids']), ids)
            with patch.object(retriever.Chroma, 'from_documents', side_effect=AssertionError('ingestion')), patch.object(
                retriever.Chroma, 'add_documents', side_effect=AssertionError('insertion')
            ), patch.object(LocalEmbeddings, 'embed_documents', side_effect=AssertionError('embedding')):
                loaded = retriever.HybridRetriever([self.a, self.b], directory)
                self.assertEqual(loaded.vectorstore._collection.count(), 2)
                self.assertEqual(set(loaded.vectorstore.get()['ids']), ids)

    def test_missing_index_does_not_create_it(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(retriever, 'OpenAIEmbeddings') as embeddings:
            with self.assertRaisesRegex(FileNotFoundError, 'build_index.py'):
                retriever.HybridRetriever([self.a], directory)
            embeddings.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_saved_chunks_only_and_app_startup_uses_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'chunks.pkl'
            with self.assertRaisesRegex(FileNotFoundError, 'build_index.py'):
                load_saved_chunks(str(path))
            self.assertFalse(path.exists())
            path.write_bytes(pickle.dumps([self.a]))
            self.assertEqual(load_saved_chunks(str(path)), [self.a])
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'src/app.py').read_text())
        startup = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'initialize_system')
        calls = [n.func.id for n in ast.walk(startup) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertIn('load_saved_chunks', calls)
        self.assertNotIn('get_or_create_chunks', calls)

    def test_duplicate_hits_cannot_outscore_cross_retriever_agreement(self):
        # Without per-list deduplication, four A contributions beat two B contributions.
        result = self.fuse([self.a, self.a, self.a, self.a, self.b], [self.b], k=5)
        self.assertEqual(result, [self.b, self.a])
        self.assertEqual(self.fuse([self.a], [self.b, self.a]), [self.a, self.b])

    def test_shared_prefix_and_different_source_page_remain_distinct(self):
        docs = [Document(page_content='x' * 100 + suffix, metadata=metadata) for suffix, metadata in [
            ('one', {'source_file': 'a.pdf', 'page': 0}),
            ('two', {'source_file': 'a.pdf', 'page': 0}),
            ('one', {'source_file': 'a.pdf', 'page': 1}),
            ('one', {'source_file': 'b.pdf', 'page': 0}),
        ]]
        self.assertEqual(len({retriever.document_id(d) for d in docs}), 4)
        self.assertEqual(self.fuse(docs, [], k=5), docs)


if __name__ == '__main__':
    unittest.main()
