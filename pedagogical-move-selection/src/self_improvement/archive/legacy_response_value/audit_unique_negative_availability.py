import json
    assert len(train_rows) == 9989
    assert len(val_rows) == 2509




    print(
        "=" * 70
    )


    print(
        "SI3-B0 — UNIQUE-NEGATIVE AVAILABILITY AUDIT"
    )


    print(
        "=" * 70
    )




    train_insufficient = audit_split(
        "SI TRAIN",
        train_rows,
    )


    val_insufficient = audit_split(
        "SI VALIDATION",
        val_rows,
    )




    print(
        "\n"
- "=" * 70
    )


    print(
        "UNIQUE-NEGATIVE AUDIT COMPLETE"
    )


    print(
        "=" * 70
    )




    print(
        "\nTrain contexts requiring exclusion:",
        len(
            train_insufficient
        )
    )


    print(
        "Validation contexts requiring exclusion:",
        len(
            val_insufficient
        )
    )




    print(
        "\nNo contrastive dataset files were written."
    )




if **name** == "**main**":
    main()
