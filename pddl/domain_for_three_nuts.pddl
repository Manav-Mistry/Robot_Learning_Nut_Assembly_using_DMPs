(define (domain NUT-ASSEMBLY)
  (:requirements :strips :typing)

  ;; Types
  (:types nut peg robot)

  ;; Predicates
  (:predicates
    (handempty ?r - robot)
    (grasped ?r - robot ?n - nut)
    (on_peg ?n - nut ?p - peg)
    (assembled ?n - nut)

    ;; Type predicates
    (is_hex ?n - nut)
    (is_square ?n - nut)
    (is_circular ?p - peg)
    (is_square_peg ?p - peg)

    ;; Compatibility
    (compatible ?n - nut ?p - peg)
  )

  ;; Pick action
  (:action pick
    :parameters (?r - robot ?n - nut)
    :precondition (handempty ?r)
    :effect (and 
      (grasped ?r ?n)
      (not (handempty ?r))
    )
  )

  ;; Generic Place action using compatibility
  (:action place
    :parameters (?r - robot ?n - nut ?p - peg)
    :precondition (and 
        (grasped ?r ?n)
        (compatible ?n ?p)
    )
    :effect (and
        (on_peg ?n ?p)
        (assembled ?n)
        (not (grasped ?r ?n))
        (handempty ?r)
    )
  )
)
