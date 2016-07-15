package lftm.utility;

public class MTRandom
{

    private static MersenneTwister rand = new MersenneTwister();

    public static void setSeed(long seed)
    {
        rand.setSeed(seed);
    }

    public static double nextDouble()
    {
//        return 0.5;
        return rand.nextDouble();
    }

    public static int nextInt(int n)
    {
        return rand.nextInt(n);
    }

    public static boolean nextBoolean()
    {
        return rand.nextBoolean();
    }
}
