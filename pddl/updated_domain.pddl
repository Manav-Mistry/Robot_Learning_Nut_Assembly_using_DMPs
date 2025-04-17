(define (domain NUT-ASSEMBLY)
  (:requirements :strips :typing)

  ;; Types
  (:types nut peg robot)

  ;; Predicates
  (:predicates
    
    (free ?p - peg)                          
    (handempty ?r - robot)                               
    (grasped ?r - robot ?n - nut)            
    (on_peg ?n - nut ?p - peg)               
    (assembled ?n - nut)                     

    
    (is_hex ?n - nut)                        
    (is_square ?n - nut)                     
    (is_circular ?p - peg)                  
    (is_square_peg ?p - peg)                  
  )

  ;; Pick action
  (:action pick
    :parameters (?r - robot ?n - nut)
    :precondition (and (handempty ?r))
    :effect (and 
      (grasped ?r ?n)
      (not (handempty ?r))
    )
  )

  ;; Place the hex nut onto a circular peg
  (:action place_on_peg_hex
    :parameters (?r - robot ?n - nut ?p - peg)
    :precondition (and 
        (grasped ?r ?n)
        (free ?p)
        (is_hex ?n)
        (is_circular ?p)
    )
    :effect (and
        (on_peg ?n ?p)
        (assembled ?n)
        (not (grasped ?r ?n))
        (handempty ?r)
        (not (free ?p))
    )
  )

  ;;place the squared nut onto a square peg
  (:action place_on_peg_square
    :parameters (?r - robot ?n - nut ?p - peg)
    :precondition (and 
        (grasped ?r ?n)
        (free ?p)
        (is_square ?n)
        (is_square_peg ?p)
    )
    :effect (and
        (on_peg ?n ?p)
        (assembled ?n)
        (not (grasped ?r ?n))
        (handempty ?r)
        (not (free ?p))
    )
  )

)
