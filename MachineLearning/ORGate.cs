using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning
{
    internal class ORGate
    {
        struct WeightedConnections
        {
            public float[] Weights;
            public float Bias;
        };

        struct NeuralNetwork
        {
            public WeightedConnections[] Connections;
        }

        static NeuralNetwork CreateNerualNetwork(params int[] dimensions)
        {
            var network = new NeuralNetwork();
            network.Connections = new WeightedConnections[dimensions.Length];
            for(int i = 0; i < dimensions.Length; i++)
            {
                var wc = new WeightedConnections();

                network.Connections[i] = wc;
            }
            return network;
        }

        static float[][] train_data = new float[][]
            {
                new float[] {0.0f, 0.0f, 0.0f, 1.0f},
                new float[] {0.0f, 0.0f, 1.0f, 1.0f},
                new float[] {0.0f, 1.0f, 0.0f, 1.0f},
                new float[] {0.0f, 1.0f, 1.0f, 0.0f},
                new float[] {1.0f, 0.0f, 0.0f, 1.0f},
                new float[] {1.0f, 0.0f, 1.0f, 1.0f},
                new float[] {1.0f, 1.0f, 0.0f, 1.0f},
                new float[] {1.0f, 1.0f, 1.0f, 0.0f},
            };

        static float evaluate(float x1, float x2, float x3, float w1, float w2, float w3, float b)
        {
            return NN.sigmoid((x1 * w1) + (x2 * w2) + (x3 * w3) + b);
        }


        static float cost(float w1, float w2, float w3, float b)
        {
            float result = 0;
            for (int i = 0; i < train_data.Length; i++)
            {
                float input1 = train_data[i][0];
                float input2 = train_data[i][1];
                float input3 = train_data[i][2];
                float expected = train_data[i][3];
                float output = evaluate(input1, input2, input3, w1, w2, w3, b);
                float c = output - expected;
                result += c * c;
            }
            result /= train_data.Length;
            return result;
        }

        static void ORMain(string[] args)
        {
            float w1 = NN.rand_float();
            float w2 = NN.rand_float();
            float w3 = NN.rand_float();
            float b = NN.rand_float();
            // b = w1 = w2 = 1.0f;
            
            float eps = 1e-3f;
            float rate = 1e-3f;

            // finite difference
            for (int i = 0; i < 25; i--)
            {
                float c = cost(w1, w2, w3, b);
                float dw1 = (cost(w1 + eps, w2, w3, b) - c) / eps;
                float dw2 = (cost(w1, w2 + eps, w3, b) - c) / eps;
                float dw3 = (cost(w1, w2, w3 + eps, b) - c) / eps;
                float db = (cost(w1, w2, w3, b + eps) - c) / eps;
                w1 -= dw1 * rate;
                w2 -= dw2 * rate;
                w3 -= dw3 * rate;
                b -= db * rate;
                Console.WriteLine($"w1 = {w1:0.0000}, w2 = {w2:0.0000}, w3 = {w3:0.0000} b = {b:0.00} --- {c}");
                Console.WriteLine($"A=0; B=0; C=0; OUT={evaluate(0, 0, 0, w1, w2, w3, b):0.0000} -> {Math.Round(evaluate(0, 0, 0, w1, w2, w3, b))}");
                Console.WriteLine($"A=0; B=0; C=1; OUT={evaluate(0, 0, 1, w1, w2, w3, b):0.0000} -> {Math.Round(evaluate(0, 0, 1, w1, w2, w3, b))}");
                Console.WriteLine($"A=0; B=1; C=0; OUT={evaluate(0, 1, 0, w1, w2, w3, b):0.0000} -> {Math.Round(evaluate(0, 1, 0, w1, w2, w3, b))}");
                Console.WriteLine($"A=0; B=1; C=1; OUT={evaluate(0, 1, 1, w1, w2, w3, b):0.0000} -> {Math.Round(evaluate(0, 1, 1, w1, w2, w3, b))}");
                Console.WriteLine($"A=1; B=0; C=0; OUT={evaluate(1, 0, 0, w1, w2, w3, b):0.0000} -> {Math.Round(evaluate(1, 0, 0, w1, w2, w3, b))}");
                Console.WriteLine($"A=1; B=0; C=1; OUT={evaluate(1, 0, 1, w1, w2, w3, b):0.0000} -> {Math.Round(evaluate(1, 0, 1, w1, w2, w3, b))}");
                Console.WriteLine($"A=1; B=1; C=0; OUT={evaluate(1, 1, 0, w1, w2, w3, b):0.0000} -> {Math.Round(evaluate(1, 1, 0, w1, w2, w3, b))}");
                Console.WriteLine($"A=1; B=1; C=1; OUT={evaluate(1, 1, 1, w1, w2, w3, b):0.0000} -> {Math.Round(evaluate(1, 1, 1, w1, w2, w3, b))}");
                Console.SetCursorPosition(0, 0);
            }


        }
    }
}
