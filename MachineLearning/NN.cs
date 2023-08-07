using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MachineLearning
{
    public sealed partial class NN
    {
        public static float rand_float() { return Random.Shared.NextSingle(); }
        public static float sigmoid(float x) { return 1.0f / (1.0f + (float)Math.Exp(-x)); }
        public static float relu(float x) { return Math.Max(0, x); }

    }
}
