(define (problem nut-assembly)
  (:domain NUT-ASSEMBLY)

  (:objects
    robot1 - robot
    hexnut squarenut - nut
    square round - peg
  )

  (:init

    ;; Pegs are free
    (free square)
    (free round)

    ;; Robot is not holding anything
    (handempty robot1)

    ;; Nut types
    (is_hex hexnut)
    (is_square squarenut)

    ;; Peg types
    (is_square_peg square)
    (is_circular round)

    ;;compatibility
    (compatible hexnut round)
    (compatible squarenut square)
  )

  (:goal
    (and
      (assembled hexnut)
      (assembled squarenut)
    )
  )
)
