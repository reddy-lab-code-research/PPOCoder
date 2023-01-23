#include <stdio.h>
#include <stdlib.h>
int flip ( int arr [ ] , int i ) {
  int temp ;
  temp = arr [ start ] ;
  arr [ start ] = arr [ i ] ;
  arr [ i ] = temp ;
  start ++ ;
  i -- ;
}
int findMax ( int arr [ ] , int n ) {
  int mi = 0 ;
  for ( int i = 0 ;
  i < n ;
  i ++ ) {
    if ( arr [ i ] > arr [ mi ] ) mi = i ;
    else mi = i ;
  }
  return mi ;
}
int pancakeSort ( int arr [ ] , int n ) {
  int curr_size = n ;
  while ( curr_size > 1 ) {
    int mi = findMax ( arr , curr_size ) ;
    if ( mi != curr_size - 1 ) flip ( arr , mi ) ;
    flip ( arr , curr_size - 1 ) ;
    curr_size -- ;
  }
}
}
void printArray ( int arr [ ] , int n ) {
  int i , temp , n ) ;
  printf ( "%d" , temp ) ;
  printf ( "Sorted_Array_" ) ;
  printArray ( arr , n ) ;
}
int main ( ) {
  int arr [ ] = {
    23 , 10 , 20 , 11 , 12 , 6 , 7 };
    int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
    pancakeSort ( arr , n ) ;
    printf ( "Sorted_Array_" ) ;
    printArray ( arr , n ) ;
    return 0 ;
  }
  