(define (domain NUT-ASSEMBLY)
  (:requirements :strips :typing)

  ;; Types
  (:types nut peg robot location)

  ;; Predicates
  (:predicates
    ;; State of the world
    (at ?n - nut ?loc - location)            ;; nut is at a location (e.g., table or peg)
    (grasped ?r - robot ?n - nut)            ;; robot is grasping the nut
    (on_peg ?n - nut ?p - peg)               ;; nut is on a peg
    (assembled ?n - nut)                     ;; nut has been successfully assembled
    (free ?p - peg)                          ;; peg is available for use
    (handempty ?r - robot)                   ;; robot is not holding anything

    ;; Nut and peg types
    (is_hex ?n - nut)                        ;; nut is hex type
    (is_square ?n - nut)                     ;; nut is square type
    (is_circular ?p - peg)                   ;; peg is circular type
    (is_square_peg ?p - peg)                 ;; peg is square type
  )

  ;; Actions

  ;; Pick action - robot picks a nut from any location
  (:action pick
    :parameters (?r - robot ?n - nut ?loc - location)
    :precondition (and (at ?n ?loc) (handempty ?r))
    :effect (and 
      (not (at ?n ?loc))
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
