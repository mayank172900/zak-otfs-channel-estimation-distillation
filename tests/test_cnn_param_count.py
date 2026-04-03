from zakotfs.cnn_model import PaperCNN


def test_cnn_parameter_count_matches_paper():
    model = PaperCNN()
    assert model.num_parameters == 245473
