from ultralytics import YOLO
import flwr as fl
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLIENT_DATA = "dataset/client3/data.yaml"

model = YOLO("yolov8n.pt")


class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):

        return [
            val.cpu().numpy()
            for _, val in model.model.state_dict().items()
        ]

    def set_parameters(self, parameters):

        params_dict = zip(
            model.model.state_dict().keys(),
            parameters
        )

        state_dict = {
            k: torch.tensor(v)
            for k, v in params_dict
        }

        model.model.load_state_dict(
            state_dict,
            strict=True
        )

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        model.train(
            data=CLIENT_DATA,
            epochs=1,
            imgsz=320,
            device=DEVICE,
            batch=2
        )

        return (
            self.get_parameters(config),
            100,
            {}
        )

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        metrics = model.val(
            data=CLIENT_DATA,
            device=DEVICE
        )

        return 0.0, 100, {
            "mAP50": float(metrics.box.map50)
        }


fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient()
)