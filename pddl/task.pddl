(define (problem nut-assembly)
  (:domain NUT-ASSEMBLY)

  (:objects
    robot1 - robot
    hexnut squarenut - nut
    peg1 peg2 - peg
  )

  (:init

    ;; Pegs are free
    (free peg1)
    (free peg2)

    ;; Robot is not holding anything
    (handempty robot1)

    ;; Nut types
    (is_hex hexnut)
    (is_square squarenut)

    ;; Peg types
    (is_square_peg peg1)
    (is_circular peg2)
  )

  (:goal
    (and
      (assembled hexnut)
      (assembled squarenut)
    )
  )
)
