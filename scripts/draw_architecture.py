"""Draw the CNN architecture and FER2013 class examples."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle, FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
INK, MUTED, RULE = "#222b34", "#526170", "#cbd3db"
CONV, CONV_SIDE = "#d7e5f1", "#b3cde2"
POOL, POOL_SIDE = "#dbece5", "#a9c9bd"
DENSE = "#e9e1d4"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                     "font.size": 14, "text.color": INK,
                     "svg.fonttype": "path", "svg.hashsalt": "fer-research-architecture",
                     "savefig.facecolor": "white"})


def label(ax, x, y, text, size=14, **kw):
    return ax.text(x, y, text, fontsize=size, ha=kw.pop("ha", "center"),
                   va=kw.pop("va", "center"), **kw)


def arrow(ax, start, end, color=INK, lw=1.2):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                mutation_scale=12, linewidth=lw, color=color))


def volume(ax, x, y, width, height, depth, color, side):
    """Front view plus a projected depth axis; depth is not proportional to C."""
    dx, dy = depth, -depth * .55
    ax.add_patch(Polygon([(x,y-height/2),(x+width,y-height/2),
                         (x+width+dx,y-height/2+dy),(x+dx,y-height/2+dy)],
                        facecolor=color, edgecolor=INK, linewidth=.65))
    ax.add_patch(Polygon([(x+width,y-height/2),(x+width+dx,y-height/2+dy),
                         (x+width+dx,y+height/2+dy),(x+width,y+height/2)],
                        facecolor=side, edgecolor=INK, linewidth=.65))
    ax.add_patch(Rectangle((x,y-height/2),width,height,
                          facecolor=color,edgecolor=INK,linewidth=.8))
    return x + width + dx


def bracket(ax, left, right, y, text):
    ax.plot([left,left,right,right],[y-4,y,y,y-4],color=RULE,lw=.9)
    label(ax,(left+right)/2,y+18,text,13)


def neuron_column(ax, x, center, n_visible=7, color=DENSE, gap=19, radius=5):
    ys=[center+(i-(n_visible-1)/2)*gap for i in range(n_visible)]
    for y in ys:
        ax.add_patch(Circle((x,y),radius,facecolor=color,edgecolor=INK,lw=.8))
    return ys


def draw_architecture():
    rows=list(csv.DictReader((ROOT/'results/architecture.csv').open()))
    conv=[r for r in rows if r['type']=='Conv2D']
    pool=[r for r in rows if r['type']=='MaxPooling2D']
    assert [r['output_shape'] for r in conv]==[
        '46 x 46 x 32','44 x 44 x 64','20 x 20 x 128',
        '18 x 18 x 128','7 x 7 x 256','5 x 5 x 256']
    assert [r['output_shape'] for r in pool]==['22 x 22 x 64','9 x 9 x 128','2 x 2 x 256']
    total=sum(int(r['parameters']) for r in rows)
    fig,ax=plt.subplots(figsize=(12.5,6.65))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set(xlim=(0,1250),ylim=(665,0));ax.axis('off')

    # (a) Paired convolutions and their pooled output, with exact H x W x C.
    label(ax,26,30,'(a) Convolutional feature extraction',18,ha='left',weight='bold')
    label(ax,26,57,r'Input $x$ → feature tensor $z$',13,ha='left',color=MUTED)
    montage=Image.open(ASSETS/'source/fer2013-training-examples.png')
    face=montage.crop((904,78,1154,328))
    ax.imshow(face,extent=(35,139,224,120),interpolation='nearest',zorder=2)
    ax.add_patch(Rectangle((35,120),104,104,facecolor='none',edgecolor=INK,lw=.8,zorder=3))
    label(ax,87,102,'FER2013 example',13)
    label(ax,87,249,r'$48\times48\times1$',15)
    label(ax,87,276,'Rescale / 255',12,color=MUTED)
    arrow(ax,(150,174),(195,174))

    # Draw six convolutional tensors, grouped into the three implemented stages.
    for stage,(left,h1,h2,hp) in enumerate([(210,116,110,70),(545,83,75,47),(880,51,43,30)]):
        c1,c2=conv[stage*2:stage*2+2]
        c1_channels=int(c1['output_shape'].split(' x ')[-1])
        c2_channels=int(c2['output_shape'].split(' x ')[-1])
        x1,x2,xp=left,left+98,left+210
        # Volume depth indicates the increasing channel count schematically.
        d1=14+stage*5;d2=19+stage*5
        e1=volume(ax,x1,174,32,h1,d1,CONV,CONV_SIDE)
        e2=volume(ax,x2,174,37,h2,d2,CONV,CONV_SIDE)
        ep=volume(ax,xp,174,22,hp,17,POOL,POOL_SIDE)
        arrow(ax,(e1+4,174),(x2-5,174));arrow(ax,(e2+4,174),(xp-5,174))
        label(ax,x1+22,97,f'{c1_channels} channels',12)
        label(ax,x2+29,97,f'{c2_channels} channels',12)
        label(ax,xp+18,113,'Pool',12,color='#386d5b')
        label(ax,x1+21,249,r'$'+c1['output_shape'].replace(' x ',r'\times')+'$',13)
        label(ax,x2+25,249,r'$'+c2['output_shape'].replace(' x ',r'\times')+'$',13)
        label(ax,xp+18,249,r'$'+pool[stage]['output_shape'].replace(' x ',r'\times')+'$',13)
        bracket(ax,left-6,xp+43,274,f'Block {stage+1}')
        if stage<2:arrow(ax,(ep+5,174),(left+330,174))
        else:
            arrow(ax,(ep+4,174),(1210,174))
            label(ax,1225,174,r'$z$',18)
    label(ax,625,323,r'Each block: [Conv $3\times3$ → ReLU → BN] × 2 → MaxPool $2\times2$ → Dropout ($p=0.25$)',14)
    ax.plot([26,1224],[351,351],color=RULE,lw=.8)

    # (b) Flattening and dense classification. Output shows class order, not scores.
    label(ax,26,382,'(b) Classification head',18,ha='left',weight='bold')
    label(ax,26,409,r'$z\in\mathbb{R}^{2\times2\times256}$ → $p\in\mathbb{R}^{7}$',13,ha='left',color=MUTED)
    volume(ax,50,511,37,45,29,POOL,POOL_SIDE)
    label(ax,84,557,r'$2\times2\times256$',14)
    label(ax,84,459,r'Feature tensor $z$',13)
    arrow(ax,(126,511),(224,511))
    label(ax,177,486,'Flatten',13)
    # A vector glyph stands for the 1,024 flattened values.
    ax.add_patch(Rectangle((236,452),24,118,facecolor='#eef1f4',edgecolor=INK,lw=.8))
    for y in [462,474,486,536,548,560]:
        ax.plot([241,255],[y,y],color=MUTED,lw=.65)
    label(ax,248,509,'⋮',19)
    label(ax,248,590,'1,024 features',14)
    arrow(ax,(269,511),(397,511))

    dense_y=neuron_column(ax,424,511,n_visible=5,gap=23)
    label(ax,424,441,'Dense 256',15,weight='bold')
    label(ax,424,582,'ReLU → BN',13)
    label(ax,424,605,r'Dropout $p=0.50$',13)
    label(ax,424,511,'⋮',17,zorder=4,
          bbox=dict(facecolor='white',edgecolor='none',pad=1))
    out_y=neuron_column(ax,696,511,n_visible=7,gap=19,color='#dbece5')
    for y1 in [dense_y[0],dense_y[1],dense_y[-2],dense_y[-1]]:
        for y2 in out_y:
            ax.plot([431,689],[y1,y2],color='#bac2cb',lw=.45,zorder=0)
    label(ax,696,421,'Dense 7',15,weight='bold')
    label(ax,559,590,'Fully connected',13,color=MUTED)
    arrow(ax,(709,511),(810,511))
    ax.add_patch(Rectangle((820,482),119,58,facecolor='#f2f5f7',edgecolor=INK,lw=.8))
    label(ax,879,511,'Softmax',16)
    arrow(ax,(950,511),(1008,511))
    # Seven actual output classes, in the generator's alphabetical order.
    classes=['angry','disgust','fear','happy','neutral','sad','surprise']
    for i,name in enumerate(classes):
        y=443+i*22
        label(ax,1030,y,rf'$p_{{{i}}}$',14,ha='left')
        label(ax,1076,y,name,14,ha='left')
    label(ax,1081,413,'Class probabilities',14,weight='bold')
    label(ax,26,646,f'{total:,} parameters  ·  BN: Batch Normalization  ·  Dimensions: H × W × C  ·  Tensor volumes are schematic.',12,ha='left',color=MUTED)
    fig.savefig(ASSETS/'model-overview.svg',metadata={'Date':None})
    plt.close(fig)


def draw_dataset_examples():
    montage=Image.open(ASSETS/'source/fer2013-training-examples.png')
    fig,axes=plt.subplots(1,7,figsize=(11,2.05))
    fig.subplots_adjust(left=.012,right=.988,top=.90,bottom=.25,wspace=.14)
    for ax,col,title in zip(axes,[0,1,2,3,6,4,5],
                            ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']):
        x=10+col*298
        ax.imshow(montage.crop((x,78,x+250,328)),interpolation='nearest')
        ax.set(xticks=[],yticks=[])
        for spine in ax.spines.values():spine.set_color(RULE);spine.set_linewidth(.6)
        ax.set_xlabel(title,fontsize=14,labelpad=9)
    fig.savefig(ASSETS/'dataset-examples.svg',metadata={'Date':None})
    plt.close(fig)


if __name__=='__main__':
    draw_architecture()
    draw_dataset_examples()
    print('Generated architecture and dataset-example figures.')
