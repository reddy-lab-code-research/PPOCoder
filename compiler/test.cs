using System ;
using System . Collections . Generic ;
public class MinSum {
  static List < int > minSqrNum ( int n ) {
    int [ ] arr = new int [ n + 1 ] ;
    int [ ] sqrNum = new int [ n + 1 ] ;
    List < int > v = new List < int > ( ) ;
    for ( int i = 0 ;
    i <= n ;
    i ++ ) {
      arr [ i ] = arr [ i - 1 ] + 1 ;
      sqrNum [ i ] = 1 ;
      int k = 1 ;
      while ( k * k <= i ) {
        if ( arr [ i ] > arr [ i - k * k ] + 1 ) {
          arr [ i ] = arr [ i - k * k ] + 1 ;
          sqrNum [ i ] = k * k ;
        }
        k ++ ;
      }
    }
    while ( n > 0 ) {
      v . Add ( sqrNum [ n ] ) ;
      n -= sqrNum [ n ] ;
    }
    return v ;
  }
  static public void Main ( String [ ] args ) {
    int n = 10 ;
    v = minSqrNum ( n ) ;
    for ( int i = 0 ;
    i < v . Count ;
    i ++ ) Console . Write ( v [ i ] ) ;
    if ( i < v . Count - 1 ) Console . Write ( "_+_" ) ;
  }
}
