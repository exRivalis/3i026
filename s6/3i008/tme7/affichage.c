#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <stdio.h>

/*
value generic_print(value a){
  CAMLparam1(a);
  if(Is_block(a)){
  	int tag = Tag_val(a);
  	if(tag < No_scan_tag){
  		printf("[%d |", tag);
  		printf(" %d,", Field(a, 0));
  		generic_print(Field(a, 1));
  		printf("]");
  	}
  	else{
  		printf("Unsupported");
  	}
  }
  else{
  	printf("%d", Int_val(a));
  }
  
  CAMLreturn(Val_unit);
}
*/
value generic_print(value a){
  CAMLparam1(a);
  if(Is_block(a)){
  	int tag = Tag_val(a);
  	if(tag < No_scan_tag){
  		printf("[%d |", tag);
  		//printf(" %d,", Int_val(Field(a, 0)));
  		generic_print(Field(a, 0));
  		generic_print(Field(a, 1));
  		printf("]");
  	}
  	else{
  		printf("[%d |", tag);
	  	switch(tag){
	  		case String_tag:
	  			printf(" %s,", String_val(Field(a, 0)));
	  			break;
	  		case Double_tag:
	  			printf(" %f,", Double_val(Field(a, 0)));
	  			break;
	  		case Double_array_tag:
	  			generic_print(Field(a, 1));
	  			break;
	  		default:
	  			printf("Unsupported");
	  			break;
	  	}
	  	
  	}
  }
  else{
  	printf(" %d,", Int_val(a));
  }
  
  CAMLreturn(Val_unit);
}
