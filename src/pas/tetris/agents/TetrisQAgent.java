package src.pas.tetris.agents;


// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.*;


// JAVA PROJECT IMPORTS
import edu.bu.tetris.agents.QAgent;
import edu.bu.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.tetris.game.Board;
import edu.bu.tetris.game.Game.GameView;
import edu.bu.tetris.game.minos.Mino;
import edu.bu.tetris.linalg.Matrix;
import edu.bu.tetris.nn.Model;
import edu.bu.tetris.nn.LossFunction;
import edu.bu.tetris.nn.Optimizer;
import edu.bu.tetris.nn.models.Sequential;
import edu.bu.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.tetris.nn.layers.ReLU;  // some activations (below too)
import edu.bu.tetris.nn.layers.Tanh;
import edu.bu.tetris.nn.layers.Sigmoid;
import edu.bu.tetris.training.data.Dataset;
import edu.bu.tetris.utils.Pair;


public class TetrisQAgent
    extends QAgent
{

    public static final double EXPLORATION_PROB = 0.05;
    public static final int BOARD_HEIGHT = 22;
    public static final int BOARD_WIDTH = 10;
    public static final int SHOULD_EXPLORE_SAMPLE_SIZE = 20;
    public static final int SHOULD_EXPLORE_MIN_MAX_MOVES_DIFF_THRESHOLD = 10;
    public static final int INPUT_MATRIX_LENGTH = 15-10;

    private Random random;
    private Matrix currentBoard = null;
    private int numWells = 0;
    private int numHoles = 0;
    private int minColMaxColDiff = 0;
    private int aggregateHeight = 0;
    private int maxHeight = 0;
    private int linesClearedWithAction = 0;

    private long minMoves = Integer.MAX_VALUE;
    private long maxMoves = Integer.MIN_VALUE;
    private long prevMoveIdx = 0;
    private long prevGameIdx = 0;
    private long prevPhaseIdx = 0;
    private boolean hasExplored = false;

    public TetrisQAgent(String name)
    {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    }

    public Random getRandom() { return this.random; }

    @Override
    public Model initQFunction()
    {
        // build a single-hidden-layer feedforward network
        // this example will create a 3-layer neural network (1 hidden layer)
        // in this example, the input to the neural network is the
        // image of the board unrolled into a giant vector
        final int inputLength = INPUT_MATRIX_LENGTH;
        final int hiddenDim = 2 * inputLength;
        final int outDim = 1;

        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(inputLength, hiddenDim));
        qFunction.add(new Tanh());
        qFunction.add(new Dense(hiddenDim, outDim));

        return qFunction;
    }

    /**
        This function is for you to figure out what your features
        are. This should end up being a single row-vector, and the
        dimensions should be what your qfunction is expecting.
        One thing we can do is get the grayscale image
        where squares in the image are 0.0 if unoccupied, 0.5 if
        there is a "background" square (i.e. that square is occupied
        but it is not the current piece being placed), and 1.0 for
        any squares that the current piece is being considered for.
        
        We can then flatten this image to get a row-vector, but we
        can do more than this! Try to be creative: how can you measure the
        "state" of the game without relying on the pixels? If you were given
        a tetris game midway through play, what properties would you look for?
     */
    @Override
    public Matrix getQFunctionInput(final GameView game,
                                    final Mino potentialAction)
    {
        Matrix finalInputMatrix = Matrix.zeros(1, INPUT_MATRIX_LENGTH);
        // columnHeights, 13 entries 
        // maxHeight, 1 entry
        // aggregateBoardHeight, 1 entry
        // numHoles, 1 entry, finds number of empty cells with a filled cell on top in each column
        // numWells, 1 entry NOT USED
        // maxColminColDiff, 1 entry

        // Matrix flattenedImage = null;
        // try
        // {
        //     flattenedImage = game.getGrayscaleImage(potentialAction).flatten();
        // } catch(Exception e)
        // {
        //     e.printStackTrace();
        //     System.exit(-1);
        // }

        Matrix boardImage = null;
        try
        {
            boardImage = game.getGrayscaleImage(potentialAction);
        } catch(Exception e)
        {
            e.printStackTrace();
            System.exit(-1);
        }
        currentBoard = boardImage;
        //System.out.println(currentBoard); //test

        List<Integer> columnHeightsList = getColumnHeights(boardImage);
        // for (int i = 0; i < columnHeightsList.size(); i++) {
        //     finalInputMatrix.set(0, i, columnHeightsList.get(i));
        // }

        aggregateHeight = columnHeightsList.stream().mapToInt(Integer::intValue).sum();
        
        Optional<Integer> maxOptional = columnHeightsList.stream().max(Integer::compareTo);
        maxHeight = maxOptional.orElse(0);
        
        Optional<Integer> minOptional = columnHeightsList.stream().min(Integer::compareTo);
        int minColumnHeight = minOptional.orElse(0);
        
        minColMaxColDiff = maxHeight - minColumnHeight;
        numHoles = getNumberOfHoles(boardImage);
        numWells = getNumberOfWells(boardImage);
        linesClearedWithAction = getLinesClearedWithAction(boardImage);
        
        finalInputMatrix.set(0, 0, aggregateHeight);
        finalInputMatrix.set(0, 1, maxHeight);
        finalInputMatrix.set(0, 2, numHoles);
        //finalInputMatrix.set(0, 13, numWells);
        finalInputMatrix.set(0, 3, minColMaxColDiff);
        finalInputMatrix.set(0, 4, linesClearedWithAction);
        //System.out.println(finalInputMatrix); // test

        return finalInputMatrix;
    }

    public List<Integer> getColumnHeights(Matrix board) {
        List<Integer> columnHeights = new ArrayList<>(10);
        for (int col = 0; col < BOARD_WIDTH; col++) {
            for (int row = 0; row < BOARD_HEIGHT; row++) {
                double cell = board.get(row, col);
                if (cell != 0.0) { 
                    columnHeights.add(BOARD_HEIGHT-row);
                    break;
                }
                if (row == BOARD_HEIGHT-1) { columnHeights.add(0); }
            }
        }
        return columnHeights;
    }

    public int getNumberOfHoles(Matrix board) {
        int numberOfHoles = 0;
        for (int col = 0; col < BOARD_WIDTH; col++) {
            boolean emptyCellFound = false;
            for (int row = BOARD_HEIGHT-1; row >= 0; row--) {
                double cell = board.get(row, col);
                if (!emptyCellFound && cell == 0.0) { 
                    emptyCellFound = true;
                }
                if (emptyCellFound && cell != 0.0) {
                    emptyCellFound = false;
                    numberOfHoles++;
                }
            }
        }
        return numberOfHoles;
    }

    public int getNumberOfWells(Matrix board) {
        int numberOfWells = 0;
        for (int i = 0; i < BOARD_WIDTH; i++) {
            for (int j = 0; j < BOARD_HEIGHT; j++) {
                double cell = board.get(i, j);
                if (cell != 0.0) { 
                    break;
                }
                if (j == BOARD_HEIGHT-1) {
                    numberOfWells++;
                }
            }
        }
        return numberOfWells;
    }

    public int getLinesClearedWithAction(Matrix board) {
        // go through each row, if a 1.0 is found, check if the entire row is occupied. if so ++
        int numLinesCleared = 0;
        for (int row = 0; row < BOARD_HEIGHT; row++) {
            for (int col = 0; col < BOARD_WIDTH; col++) {
                double cell = board.get(row, col);
                if (cell == 1.0) { // cell being considered
                    boolean isFullLine = true;
                    for (int col1 = 0; col1 < BOARD_WIDTH; col1++) {
                        double cell1 = board.get(row, col1);
                        if (cell1 == 0.0) {
                            isFullLine = false;
                            break;
                        }
                    } 
                    if (isFullLine) {
                        numLinesCleared++;
                    }
                }
            }
        }
        return numLinesCleared;
    }

    /**
     * This method is used to decide if we should follow our current policy
     * (i.e. our q-function), or if we should ignore it and take a random action
     * (i.e. explore).
     *
     * Remember, as the q-function learns, it will start to predict the same "good" actions
     * over and over again. This can prevent us from discovering new, potentially even
     * better states, which we want to do! So, sometimes we should ignore our policy
     * and explore to gain novel experiences.
     *
     * The current implementation chooses to ignore the current policy around 5% of the time.
     * While this strategy is easy to implement, it often doesn't perform well and is
     * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
     * strategy here.
     */
    @Override
    public boolean shouldExplore(final GameView game,
                                 final GameCounter gameCounter)
    {
        // every 20 games, check the minimum moves and maximum moves. 
        // if more than 10, consider the model is "stuck" and return true
        if (gameCounter.getCurrentPhaseIdx() != prevPhaseIdx) {
            prevMoveIdx = 0;
            prevGameIdx = 0;
            prevPhaseIdx = gameCounter.getCurrentPhaseIdx();
            return false; 
        }
        if (gameCounter.getCurrentGameIdx() != prevGameIdx) { // signifies the current game ended
            if (prevMoveIdx < minMoves) { minMoves = prevMoveIdx; }
            if (prevMoveIdx > maxMoves) { maxMoves = prevMoveIdx; }
        }
        if (gameCounter.getCurrentGameIdx() % SHOULD_EXPLORE_SAMPLE_SIZE == 0) { // after 20 games
            if ((maxMoves - minMoves) > SHOULD_EXPLORE_MIN_MAX_MOVES_DIFF_THRESHOLD && !hasExplored) {
                //System.out.println(true); // test
                hasExplored = true;
                return true;
            }
            minMoves = 0;
            maxMoves = 0; 
        } else {
            hasExplored = false;
        }
        prevMoveIdx = gameCounter.getCurrentMoveIdx();
        prevGameIdx = gameCounter.getCurrentGameIdx();
        prevPhaseIdx = gameCounter.getCurrentPhaseIdx();
        //System.out.println(false); // test
        return false;
    }

    /**
     * This method is a counterpart to the "shouldExplore" method. Whenever we decide
     * that we should ignore our policy, we now have to actually choose an action.
     *
     * You should come up with a way of choosing an action so that the model gets
     * to experience something new. The current implemention just chooses a random
     * option, which in practice doesn't work as well as a more guided strategy.
     * I would recommend devising your own strategy here.
     */
    @Override
    public Mino getExplorationMove(final GameView game)
    {
        int randIdx = this.getRandom().nextInt(game.getFinalMinoPositions().size());
        return game.getFinalMinoPositions().get(randIdx);
    }

    /**
     * This method is called by the TrainerAgent after we have played enough training games.
     * In between the training section and the evaluation section of a phase, we need to use
     * the exprience we've collected (from the training games) to improve the q-function.
     *
     * You don't really need to change this method unless you want to. All that happens
     * is that we will use the experiences currently stored in the replay buffer to update
     * our model. Updates (i.e. gradient descent updates) will be applied per minibatch
     * (i.e. a subset of the entire dataset) rather than in a vanilla gradient descent manner
     * (i.e. all at once)...this often works better and is an active area of research.
     *
     * Each pass through the data is called an epoch, and we will perform "numUpdates" amount
     * of epochs in between the training and eval sections of each phase.
     */
    @Override
    public void trainQFunction(Dataset dataset,
                               LossFunction lossFunction,
                               Optimizer optimizer,
                               long numUpdates)
    {
        for(int epochIdx = 0; epochIdx < numUpdates; ++epochIdx)
        {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix> > batchIterator = dataset.iterator();

            while(batchIterator.hasNext())
            {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try
                {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(),
                                                  lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch(Exception e)
                {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }

    /**
     * This method is where you will devise your own reward signal. Remember, the larger
     * the number, the more "pleasurable" it is to the model, and the smaller the number,
     * the more "painful" to the model.
     *
     * This is where you get to tell the model how "good" or "bad" the game is.
     * Since you earn points in this game, the reward should probably be influenced by the
     * points, however this is not all. In fact, just using the points earned this turn
     * is a **terrible** reward function, because earning points is hard!!
     *
     * I would recommend you to consider other ways of measuring "good"ness and "bad"ness
     * of the game. For instance, the higher the stack of minos gets....generally the worse
     * (unless you have a long hole waiting for an I-block). When you design a reward
     * signal that is less sparse, you should see your model optimize this reward over time.
     */
    @Override
    public double getReward(final GameView game)
    {
        // punish having wells
        // punish holes
        // punish large diff between min col and max col
        // punish large aggregate height
        // punish max height    
        double finalReward = 0.7*game.getScoreThisTurn() - 0.1*numHoles - 0.05*minColMaxColDiff - 0.1*aggregateHeight - 0.05*maxHeight;
        return finalReward;
    }

}
