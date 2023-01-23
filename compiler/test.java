import java . io . * ;
class GFG {
  static int countSubarray ( int arr [ ] , int K , int N ) {
    if ( K % 2 != 0 ) return 0 ;
    if ( N < K ) return 0 ;
    int start = 0 ;
    int i = 0 ;
    int count = 0 ;
    int currXor = arr [ i ] ;
    i ++ ;
    while ( i < K ) {
      currXor ^= arr [ i ] ;
      i ++ ;
    }
    if ( currXor == 0 ) count ++ ;
    currXor ^= arr [ start ] ;
    start ++ ;
    while ( i < N ) {
      currXor ^= arr [ i ] ;
      i ++ ;
      if ( currXor == 0 ) count ++ ;
      currXor ^= arr [ start ] ;
      start ++ ;
    }
    return count ;
  }
  public static void main ( String args [ ] ) {
    int arr [ ] = {
      2 , 4 , 4 , 2 , 2 , 4 };
      int K = 4 ;
      int N = arr . length ;
      System . out . print ( countSubarray ( arr , K , N ) ) ;
    }
  }
  