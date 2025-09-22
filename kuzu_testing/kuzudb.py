import kuzu

def main():
    # Create an empty on-disk database and connect to it
    db = kuzu.Database("example.kuzu")
    conn = kuzu.Connection(db)

    # Create schema
    conn.execute("CREATE NODE TABLE User(name STRING PRIMARY KEY, age INT64)")
    conn.execute("CREATE NODE TABLE City(name STRING PRIMARY KEY, population INT64)")
    conn.execute("CREATE REL TABLE Follows(FROM User TO User, since INT64)")
    conn.execute("CREATE REL TABLE LivesIn(FROM User TO City)")

    # Insert data
    conn.execute('COPY User FROM "./data/user.csv"')
    conn.execute('COPY City FROM "./data/city.csv"')
    conn.execute('COPY Follows FROM "./data/follows.csv"')
    conn.execute('COPY LivesIn FROM "./data/lives-in.csv"')

    # Execute Cypher query
    response = conn.execute(
        """
        MATCH (a:User)-[f:Follows]->(b:User)
        RETURN a.name, b.name, f.since;
        """
    )
    for row in response:
        print(row)
    # response = conn.execute(
    # """
    # MATCH (a:User)-[f:Follows]->(b:User)
    # RETURN a.name, b.name, f.since;
    # """
    # )
    # for row in response.rows_as_dict():
    #     print(row)

if __name__ == "__main__":
    main()