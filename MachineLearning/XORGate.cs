using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning;
public class XORGate
{
    struct Xor
    {
        public Matrix input;
        public Matrix w1;
        public Matrix w2;
        public Matrix b1;
        public Matrix b2;
        public Matrix a1;
        public Matrix a2;
    }

    static float forward_xor(Xor xor)
    {
        Matrix.Dot(xor.a1, xor.input, xor.w1);
        Matrix.Sum(xor.a1, xor.b1);
        Matrix.Sigmoid(xor.a1);

        Matrix.Dot(xor.a2, xor.a1, xor.w2);
        Matrix.Sum(xor.a2, xor.b2);
        Matrix.Sigmoid(xor.a2);
        
        return xor.a2.data[0, 0];
    }

    static float cost(Xor m, Matrix trainData, Matrix trainOutput)
    {
        float result = 0;
        if (trainData.rows != trainOutput.rows) throw new InvalidOperationException("Mismatch detected.");
        if (trainOutput.columns != m.a2.columns) throw new InvalidOperationException("Mismatch detected.");
        long n = trainOutput.rows;
        long col = trainOutput.columns;
        for(long i = 0; i < n; i++)
        {
            Matrix x = Matrix.Row(trainData, i);
            Matrix y = Matrix.Row(trainOutput, i);
            Matrix.Copy(m.input, x);
            forward_xor(m);
            // compute differences
            for(int j = 0; j < col; j++)
            {
                float d = m.a2.data[0, j] - y.data[0, j];
                result += d * d;
            }
        }

        return result/n;
    }

    static void finite_diff(Xor m, Xor g, float eps, Matrix trainInput, Matrix trainOutput)
    {
        float c0 = cost(m, trainInput, trainOutput);
        for (int r = 0; r < m.w1.rows; r++)
        {
            float saved;
            for (int c = 0; c < m.w1.columns; c++)
            {
                // save bits
                saved = m.w1.data[r, c];
                m.w1.data[r, c] += eps;
                float diff = (cost(m, trainInput, trainOutput) - c0) / eps;
                g.w1.data[r, c] = diff;
                // restore bits
                m.w1.data[r, c] = saved;
            }
        }

        for (int r = 0; r < m.b1.rows; r++)
        {
            float saved;
            for (int c = 0; c < m.b1.columns; c++)
            {
                // save bits
                saved = m.b1.data[r, c];
                m.b1.data[r, c] += eps;
                float diff = (cost(m, trainInput, trainOutput) - c0) / eps;
                g.b1.data[r, c] = diff;
                // restore bits
                m.b1.data[r, c] = saved;
            }
        }

        for (int r = 0; r < m.w2.rows; r++)
        {
            float saved;
            for (int c = 0; c < m.w2.columns; c++)
            {
                // save bits
                saved = m.w2.data[r, c];
                m.w2.data[r, c] += eps;
                float diff = (cost(m, trainInput, trainOutput) - c0) / eps;
                g.w2.data[r, c] = diff;
                // restore bits
                m.w2.data[r, c] = saved;
            }
        }

        for (int r = 0; r < m.b2.rows; r++)
        {
            float saved;
            for (int c = 0; c < m.b2.columns; c++)
            {
                // save bits
                saved = m.b2.data[r, c];
                m.b2.data[r, c] += eps;
                float diff = (cost(m, trainInput, trainOutput) - c0) / eps;
                g.b2.data[r, c] = diff;
                // restore bits
                m.b2.data[r, c] = saved;
            }
        }

    }

    static void apply_gradient(Xor m, Xor g, float rate)
    {
        for (int r = 0; r < m.w1.rows; r++)
        {
            for (int c = 0; c < m.w1.columns; c++)
            {
                m.w1.data[r, c] -= g.w1.data[r, c] * rate;
            }
        }
        for (int r = 0; r < m.b1.rows; r++)
        {
            for (int c = 0; c < m.b1.columns; c++)
            {
                m.b1.data[r, c] -= g.b1.data[r, c] * rate;
            }
        }
        for (int r = 0; r < m.w2.rows; r++)
        {
            for (int c = 0; c < m.w2.columns; c++)
            {
                m.w2.data[r, c] -= g.w2.data[r, c] * rate;
            }
        }
        for (int r = 0; r < m.b2.rows; r++)
        {
            for (int c = 0; c < m.b2.columns; c++)
            {
                m.b2.data[r, c] -= g.b2.data[r, c] * rate;
            }
        }
    }

    static void Main(string[] args)
    {
        Xor xor = new()
        {
            input = new(1, 2),
            w1 = new(2, 2),
            a1 = new(1, 2),
            b1 = new(1, 2),
            w2 = new(2, 1),
            a2 = new(1, 1),
            b2 = new(1, 1)
        };

        Xor grad = new()
        {
            input = new(1, 2),
            w1 = new(2, 2),
            a1 = new(1, 2),
            b1 = new(1, 2),
            w2 = new(2, 1),
            a2 = new(1, 1),
            b2 = new(1, 1)
        };

        xor.w1.FillRandom(0.0f, 1.0f);
        xor.w2.FillRandom(0.0f, 1.0f);
        xor.b1.FillRandom(0.0f, 1.0f);
        xor.b2.FillRandom(0.0f, 1.0f);

        

        Matrix trainInput = new Matrix(4, 2);
        trainInput.data[0, 0] = 0;
        trainInput.data[0, 1] = 0;
        trainInput.data[1, 0] = 0;
        trainInput.data[1, 1] = 1;
        trainInput.data[2, 0] = 1;
        trainInput.data[2, 1] = 0;
        trainInput.data[3, 0] = 1;
        trainInput.data[3, 1] = 1;

        Matrix trainOutput = new Matrix(4, 1);
        trainOutput.data[0, 0] = 0;
        trainOutput.data[1, 0] = 1;
        trainOutput.data[2, 0] = 1;
        trainOutput.data[3, 0] = 0;

        while (true)
        {
            float c = cost(xor, trainInput, trainOutput);
            Console.WriteLine($"cost = {c:0.0000}");
            for (int j = 0; j <= 1; j++)
            {
                for (int k = 0; k <= 1; k++)
                {
                    xor.input.data[0, 0] = j;
                    xor.input.data[0, 1] = k;
                    Console.WriteLine($"{j} ^ {k} = {forward_xor(xor):0.0000} -> {Math.Round(forward_xor(xor))}");
                }
            }
            xor.w1.Print("w1");
            xor.w2.Print("w2");
            xor.b1.Print("b1");
            xor.b2.Print("b2");
            Console.SetCursorPosition(0, 0);
            // LEARNING
            finite_diff(xor, grad, 1e-3f, trainInput, trainOutput);
            apply_gradient(xor, grad, 1e-2f);
        }

    }

}

