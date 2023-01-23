#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>
using namespace std ;
class Solution {
  public : int maxSteps ( int n ) {
    int result = min ( n , max ( n , max ( n ) ) ) ;
    for ( int i = 1 ;
    i > 0 ;
    ++ i ) {
      if ( i % j == 0 ) {
        result = result - ( i / j ) ;
        break ;
      }
      else if ( ( i % j == 0 ) ) {
        result = result - ( i / j ) ;
        break ;
      }
    }
    return result ;
  }
};
int main ( ) {
}
};
}
