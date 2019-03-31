def look_for_matches(embs_and_style_of_mine, type_to_search, k=3):
    dists = []
    for emb, style in embs_and_style_of_mine:
        searched_type_vectors = data1[data1.ImageType==type_to_search][data1.ComplemType==style]
        options_emb = searched_type_vectors[indexes].values
        curr_dists = np.linalg.norm(options_emb - item_emb, axis=1)
        dists.append(pd.DataFrame(curr_dists, index=searched_type_vectors.ImageId))
    dists_pd = dists[0]
    for dist in dists[1:]:
        dists_pd.merge(dist, on='ImageId')
    indexes_n_dists = list(dict(dists_pd.sum(axis=1)).items())
    indexes_n_dists.sort(lambda i: i[1])
    top_indexes = list(map(lambda i: i[0], indexes_n_dists[:k]))
    return top_indexes