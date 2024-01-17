from functions_preprocess_graph import PreprocessTabular

if __name__ == "__main__":   
    train_data = read_train_data()
    test_data = read_test_data()
    pipeline = Pipeline([
        ('PreprocessTabular', CreateOurDataset("smiles", "mol", 8)),
        # Add more pipeline steps if needed
    ])
    result_dataset = pipeline.fit_transform(train_data)
