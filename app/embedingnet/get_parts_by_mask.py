old_codes = [
 'null',
 'bag',
 'belt',
 'boots',
 'footwear',
 'outer',
 'dress',
 'sunglasses',
 'pants',
 'top',
 'shorts',
 'skirt',
 'headwear',
 'scarf & tie']

new_codes = ['tops',
 'shoes',
 'all-body',
 'scarves',
 'outerwear',
 'accessories',
 'hats',
 'bags',
 'bottoms',
 'sunglasses']

matches = {
 'bag':'bags',
 'belt':'accessories',
 'boots':'shoes',
 'footwear':'shoes',
 'outer':'outerwear',
 'dress':'all-body',
 'sunglasses':'sunglasses',
 'pants':'bottoms',
 'top':'tops',
 'shorts':'bottoms',
 'skirt':'bottoms',
 'headwear':'hats',
 'scarf & tie':'scarves'}

old_code_to_new_and_name = {i + 1 : (new_codes.index(matches[name]) + 1, matches[name]) for i, name in enumerate(old_codes[1:])}

def get_parts_by_mask(img, mask):
    for key, value in old_code_to_new_and_name.items():
        mask[mask==key] = value[0]
    classes = set(mask.flatten())
    resized_mask = np.round(np.array(Image.fromarray(np.uint8(mask / max(classes) * 255), 'L').resize(reversed(img.shape[:2]))) / 255 * max(classes))
    classes = list(set(mask.flatten()))
    if 0 in classes:
        classes.remove(0)
    
    final_imgs = []
    for cls in classes:
        curr_mask = (resized_mask == cls) * 1.
        new_img = np.zeros(img.shape)
        new_img[:, :, 0] = np.multiply(img[:, :, 0], curr_mask)
        new_img[:, :, 1] = np.multiply(img[:, :, 1], curr_mask)
        new_img[:, :, 2] = np.multiply(img[:, :, 2], curr_mask)
        new_img[new_img==0] = 255.
        new_img = new_img.astype(np.uint8)
        axis0 = np.where(curr_mask.sum(axis=0) > 0)[0]
        axis1 = np.where(curr_mask.sum(axis=1) > 0)[0]
        min_x, max_x = min(axis0), max(axis0)
        min_y, max_y = min(axis1), max(axis1)
        new_img = new_img[min_y : max_y, min_x : max_x]
        N, M, _ = new_img.shape 
        d = int(max(N, M) / 10)
        D = max(N, M) + 2 * d
        img_to_reshape = np.ones((D, D, 3)) * 255.
        if N > M:
            y = d + int((D - M) / 2)
            img_to_reshape[d:N + d, y - d:y - d + M, :] = new_img
        else:
            x = d + int((D - N) / 2)
            img_to_reshape[x - d : x - d + N, d : M + d] = new_img
        img_to_reshape = img_to_reshape.astype(np.uint8)
        sized_img = Image.fromarray(img_to_reshape).resize((112, 112))
        final_imgs.append((sized_img, new_codes[cls - 1]))
    return final_imgs