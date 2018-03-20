external generic_print : 'a -> unit = "generic_print"

let a = Array.make 5 1

let () =
  generic_print(2.5);
  print_newline ()
   
