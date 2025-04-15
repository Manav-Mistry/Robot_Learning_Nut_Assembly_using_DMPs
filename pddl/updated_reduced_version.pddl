(define (domain NUT-ASSEMBLY)
  (:requirements :strips :typing)

  ;; Types
  (:types nut peg robot location)

  ;; Predicates
  (:predicates
    ;; World state
    (at ?n - nut ?loc - location)            
    (assembled ?n - nut)                     
    (free ?p - peg)                          
    (handempty ?r - robot)                   

    ;; Nut and peg types
    (is_hex ?n - nut)                        
    (is_square ?n - nut)                     
    (is_circular ?p - peg)                   
    (is_square_peg ?p - peg)                 
  )

  ;; ACTION: Assemble hex nut (pick + place in one)
  (:action assemble_hex_nut
    :parameters (?r - robot ?n - nut ?p - peg ?loc - location)
    :precondition (and 
      (at ?n ?loc)
      (handempty ?r)
      (free ?p)
      (is_hex ?n)
      (is_circular ?p)
    )
    :effect (and
      (not (at ?n ?loc))
      (assembled ?n)
      (not (free ?p))
    )
  )

  ;; ACTION: Assemble square nut (pick + place in one)
  (:action assemble_square_nut
    :parameters (?r - robot ?n - nut ?p - peg ?loc - location)
    :precondition (and 
      (at ?n ?loc)
      (handempty ?r)
      (free ?p)
      (is_square ?n)
      (is_square_peg ?p)
    )
    :effect (and
      (not (at ?n ?loc))
      (assembled ?n)
      (not (free ?p))
    )
  )
)
