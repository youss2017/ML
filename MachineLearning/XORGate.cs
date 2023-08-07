using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning;
public class XORGate
{
    static void Main(string[] args)
    {
        Matrix w1 = new(2, 2);
        Matrix b1 = new(1, 2);
        Matrix w2 = new(2, 1);
        Matrix b2 = new(1, 1);

        w1.FillRandom(0.0f, 1.0f);
        w2.FillRandom(0.0f, 1.0f);
        b1.FillRandom(0.0f, 1.0f);
        b2.FillRandom(0.0f, 1.0f);

        w1.Print("w1");
        w2.Print("w2");
        b1.Print("b1");
        b2.Print("b2");

    }

}

