using System.Text;

namespace MachineLearning
{
    public struct Matrix
    {
        public long rows;
        public long columns;
        public float[,] data;

        public Matrix(long rows, long columns)
        {
            this.rows = rows;
            this.columns = columns;
            data = new float[rows, columns];
        }

        public static void Copy(Matrix dst, Matrix src)
        {
            if (dst.columns != src.columns) throw new InvalidOperationException("Mismatch detected.");
            if (dst.rows != src.rows) throw new InvalidOperationException("Mismatch detected.");
            for(int r = 0; r < src.rows; r++)
            {
                for(int c = 0; c < src.columns; c++)
                {
                    dst.data[r, c] = src.data[r, c];
                }
            }
        }

        public static Matrix Row(Matrix m, long row)
        {
            Matrix result = new Matrix(1, m.columns);
            for(int i = 0; i < m.columns; i++)
            {
                result.data[0, i] = m.data[row, i];
            }
            return result;
        }

        public void Identity()
        {
            long k = Math.Min(rows, columns);
            for (long i = 0; i < k; i++)
                data[i, i] = 1.0f;
        }

        public void Fill(float value)
        {
            for (long r = 0; r < rows; r++)
            {
                for (long c = 0; c < columns; c++)
                {
                    data[r, c] = value;
                }
            }
        }

        public void FillRandom(float low, float high)
        {
            for (long r = 0; r < rows; r++)
            {
                for (long c = 0; c < columns; c++)
                {
                    data[r, c] = Random.Shared.NextSingle() * (high - low) + low;
                }
            }
        }

        public static void Sigmoid(Matrix dst)
        {
            for (long r = 0; r < dst.rows; r++)
            {
                for (long c = 0; c < dst.columns; c++)
                {
                    dst.data[r, c] = NN.sigmoid(dst.data[r, c]);
                }
            }
        }

        public override string ToString()
        {
            StringBuilder str = new();
            const string fmt = " 0.00;-0.00";
            for (long r = 0; r < rows; r++)
            {
                str.Append("[ ");
                for (long c = 0; c < columns; c++)
                {
                    str.Append(data[r, c].ToString(fmt));
                    str.Append(' ');
                }
                str.Append(']');
                if (r < rows - 1)
                    str.Append('\n');
            }
            return str.ToString();
        }

        public static void Dot(Matrix dst, Matrix a, Matrix b)
        {
            if (dst.rows != a.rows || dst.columns != b.columns)
            {
                throw new InvalidOperationException("Cannot perform dot product because dst dimensions are invalid.");
            }
            if (a.columns != b.rows)
            {
                throw new InvalidOperationException("Cannot perform dot product because matrix A and B do not match.");
            }
            long n = a.columns;
            for(long r = 0; r < dst.rows; r++)
            {
                for(long c = 0; c < dst.columns; c++)
                {
                    for(long k = 0; k < n; k++)
                    {
                        dst.data[r, c] += a.data[r, k] * b.data[k, c];
                    }
                }
            }
        }

        public static void Sum(Matrix dst, Matrix a)
        {
            if (dst.rows != a.rows || dst.columns != a.columns)
            {
                throw new InvalidOperationException("Cannot perform sum operation because DST dimensions are not equal to A dimensions.");
            }
            for (long r = 0; r < dst.rows; r++)
            {
                for (long c = 0; c < dst.columns; c++)
                {
                    dst.data[r, c] += a.data[r, c];
                }
            }
        }

        public void Print(string name = "")
        {
            if(name.Length > 0)
            {
                Console.WriteLine($"{name} = ");
            }
            Console.WriteLine(ToString());
        }

    }

}
