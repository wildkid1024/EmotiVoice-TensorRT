# EmotiVoice-TensorRT: The faster EmotiVoice infer engine with ~8x speedup 

We optimized the EmotiVoice inference engine with ~8x speedup.
The speedup framework we are used is TensorRT 8.x and above.


## How to Run

Zero, you should update the EmotiVoice submodule:

```bash
git submodule init
git submodule update
```

First, convert the torch model to onnx model:
```
python th2onnx.py
```

Then, you will get the onnx model in `outputs/onnx` dir.
Just generate the .trt file for TensorRT inference:

```
python onnx2trt.py
```

Finally, just try it in a TTS application:
```
python main.py
```
Or run with openai http api:
```
python openai_api.py
```

Just enjoy it, good luck!

## Benchmark

We tests our optimization on a single nvidia 2070's GPU card.

| model | torch | ours| speedup |
| -- | -- | -- | -- 
| backend | 85 ms | 11 ms| ~8x
| front+backend | 967ms | 867 ms | ~10%

We suggest @Netease to optimize the frontend.

Maybe a cpp frontend is useful when you deploy in the production environments.


## References

If this work is helpful for you, please star it.



## License

EmotiVoice-TensorRT is provided under the Apache-2.0 License - see the LICENSE file for details.

