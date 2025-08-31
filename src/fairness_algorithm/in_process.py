from src.models.baseline_models import MFModel, LightGCNModel

def apply_in_process(model, train_df, val_df=None):
    # model.fit already applies in_method
    model.fit(train_df, val_df)
    predictions = model.predict(train_df)
    return predictions