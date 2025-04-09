(define (problem nut-assembly-problem)
  (:domain NUT-ASSEMBLY)

  (:objects
    ;; Objects by type
    robot1 - robot
    hexnut1 squarenut1 - nut
    peg1 peg2 - peg
    table - location
  )

  (:init
    ;; Nut initial positions
    (at hexnut1 table)
    (at squarenut1 table)

    ;; Pegs are free
    (free peg1)
    (free peg2)

    ;; Robot is not holding anything
    (handempty robot1)

    ;; Nut types
    (is_hex hexnut1)
    (is_square squarenut1)

    ;; Peg types
    (is_circular peg1)
    (is_square_peg peg2)
  )

  (:goal
    (and
      (assembled hexnut1)
      (assembled squarenut1)
    )
  )
)
