(define (problem nut-assembly)
  (:domain NUT-ASSEMBLY)

  (:objects
    robot1 - robot
    hexnut1 hexnut2 squarenut  - nut
    square round - peg
  )

  (:init

    ;; Robot is not holding anything
    (handempty robot1)

    ;; Nut types
    (is_hex hexnut1)
    (is_hex hexnut2)
    (is_square squarenut)

    ;; Peg types
    (is_square_peg square)
    (is_circular round)

    ;;compatibility
    (compatible hexnut1 round)
    (compatible hexnut2 round)
    (compatible squarenut square)
  )

  (:goal
    (and
      (assembled hexnut2)
      (assembled squarenut)
      (assembled hexnut1)
    )
  )
)
