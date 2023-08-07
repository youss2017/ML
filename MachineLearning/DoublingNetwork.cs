namespace MachineLearning
{
    internal class DoublingNetwork
    {
        static float rand_float() {  return Random.Shared.NextSingle(); }

        static float[][] train_data = new float[][]
            {
                new float[] {1.0f, 1.0f},
                new float[] {2.0f, 4.0f},
                new float[] {3.0f, 6.0f},
                new float[] {4.0f, 8.0f},
            };

        static float cost(float w, float b)
        {
            float result = 0;
            for(int i = 0; i < train_data.Length; i++)
            {
                float input = train_data[i][0];
                float expected = train_data[i][1];
                float output = input * w + b;
                float c = (output - expected);
                result += c * c;
            }
            result /= train_data.Length;
            return result;
        }

        static void DoublingNetworkMain(string[] args)
        {
            float w = rand_float() * 150.0f;
            float b = rand_float() * 50.0f;

            float eps = 1e-3f;
            float rate = 1e-3f;

            Console.WriteLine($"w = {w}");
            Console.WriteLine("----------------------------");
            for(int i = 0; i < 10000; i++)
            {
                float c = cost(w, b);
                Console.Write($"Cost = {c:0.0000} --- {i}\r");
                float dcost_w = (cost(w + eps, b) - cost(w, b)) / eps;
                float dcost_b = (cost(w, b + eps) - cost(w, b)) / eps;
                w -= dcost_w * rate;
                b -= dcost_b * rate;
                Thread.Sleep(6);
            }
            Console.WriteLine("\n----------------------------");
            Console.WriteLine($"w = {w} --- b = {b}");
        }

    }
}