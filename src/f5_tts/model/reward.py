import torch
import torchaudio
from fastapi import FastAPI
from pydantic import BaseModel
from vocos import Vocos
import utmosv2
from f5_tts.infer.utils_infer import load_vocoder

class LocalMOSPipeline:
    def __init__(self, vocos_path, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

        # Load Vocos (24k)
        config_path = f"{vocos_path}/config.yaml"
        model_path = f"{vocos_path}/pytorch_model.bin"

        self.vocoder = load_vocoder(vocoder_name = "vocos", is_local= False
            # vocoder_name=vocoder_name, is_local=load_vocoder_from_local, local_path=vocoder_local_path, device=device
        )
        self.vocoder = self.vocoder.eval().to(self.device)

        # Load UTMOS
        self.utmos = utmosv2.create_model(pretrained=True).to(self.device)
        self.utmos.eval()

        # Resample 24k -> 16k
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=24000,
            new_freq=16000
        )

    @torch.no_grad()
    def infer_from_mel(self, mel, ref_len):

        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        mel = mel.to(self.device)                # (B, T, C)

        mel = mel[:, ref_len:, :]                # remove reference segment

        mel = mel.permute(0, 2, 1).contiguous()  # (B, C, T)

        waveform = self.vocoder.decode(mel)      # F5 vocos wrapper

        if waveform.dim() == 2 and waveform.size(0) == 1:
            waveform = waveform.squeeze(0)

        waveform = waveform.float()

        waveform_16k = self.resampler(waveform.cpu())

        mos = self.utmos.predict(
            data=waveform_16k,
            sr=16000
        )

        return float(mos)



app = FastAPI()

pipeline = LocalMOSPipeline(
    vocos_path="/path/to/vocos-mel-24khz",
    device="cuda"
)

class MelRequest(BaseModel):
    mel: list
    ref_len: int

@app.post("/infer")
async def infer(req: MelRequest):

    mel_tensor = torch.tensor(req.mel).float().unsqueeze(0)

    mos = pipeline.infer_from_mel(
        mel_tensor,
        ref_len=req.ref_len
    )

    return {"mos": mos}

# uvicorn reward:app --host 0.0.0.0 --port 8000