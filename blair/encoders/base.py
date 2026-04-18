class BaseSemanticEncoder:
    def __init__(self, model, gpu_id, batch_size, **kwargs):
        self.model = model
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        # Keep PCA parameters for backward compatibility, but deprecate their use
        self.pca = kwargs.get('pca', False)
        self.n_comps = kwargs.get('n_comps', 0)
        self.whiten = kwargs.get('whiten', False)

    @property
    def name(self):
        raise NotImplementedError()

    def encode(self, sentences):
        raise NotImplementedError()
    
    def pca_whiten_embeddings(self, embeddings):
        """
        Deprecated: Use blair.utils.apply_pca_if_needed for global PCA post-processing.
        This method is kept for backward compatibility only.
        """
        print("WARNING: pca_whiten_embeddings is deprecated. Use blair.utils.apply_pca_if_needed instead.")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.n_comps, whiten=self.whiten)
        return pca.fit_transform(embeddings)
    
    def get_pca_config(self):
        """Get PCA configuration as a dict for the new PCA system."""
        return {
            'enabled': self.pca,
            'n_components': self.n_comps,
            'whiten': self.whiten
        }
