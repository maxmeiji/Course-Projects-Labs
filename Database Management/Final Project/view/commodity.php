
<?php

require_once '../db.php';



@session_start();
if(!isset($_SESSION['is_login']) || !$_SESSION['is_login'])
{
	//直接轉跳到 index.php
	header("Location: /final/list.php");
}
require_once '../login_status.php';
?>
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
        <meta name="description" content="" />
        <meta name="author" content="" />
        <title>商品資訊頁</title>
        
        <link rel="icon" type="image/x-icon" href="assets/favicon.ico" />
        
        <script src="https://use.fontawesome.com/releases/v5.15.3/js/all.js" crossorigin="anonymous"></script>
        
        <link href="final.css" rel="stylesheet" />
    </head>
    <body>
        
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <div class="container">
                <a class="navbar-brand" href="final2.php">YAPPI Shop</a>
                <button class="navbar-toggler navbar-toggler-right" type="button" data-toggle="collapse" data-target="#navbarResponsive" aria-controls="navbarResponsive" aria-expanded="false" aria-label="Toggle navigation"><span class="navbar-toggler-icon"></span></button>
                <div class="collapse navbar-collapse" id="navbarResponsive">
                    <ul class="navbar-nav ml-auto">
                        <li class="nav-item"><a class="nav-link" href="cartnew2.php">結帳去</a></li>
                        <li class="nav-item"><a class="nav-link" href="cartnew2.php">購物車</a></li>
                        
                        
                    </ul>
                </div>
            </div>
        </nav>
        
        <section class="py-5">
            <div class="container">
                
                <h1>商品列表<!--可將商品列表改成商品名稱--></h1>
                <ol class="breadcrumb mb-4">
                    <li class="breadcrumb-item"><a href="final2.php">首頁</a></li>
                    <li class="breadcrumb-item active">商品列表</li>
                </ol>
                
                <div class="row">
				<?php
							if(!isset($_GET['c_id'])){
								//沒有pid則直接跳轉回products.php
								header("location: final2.php");
							}else{
								$c_id=$_GET['c_id'];
							}
							?>
							<?php
							 $link = mysqli_connect($host,$dbuser,$dbpw,$dbname) or die ("-1");
							  mysqli_select_db($link,$dbname) or die("cannot connect to db");
							  $query = "SELECT * FROM commodity where c_id=$c_id ORDER BY c_id DESC ";
							  $result = mysqli_query($link,$query) or die("cannot connect to table" . mysqli_error( ));
							?>
							<?php
							while($rs = mysqli_fetch_array($result))
								{
									
							?>
                    <div class="col-lg-6"><div class="img"><?php echo '<p><img src="',$rs['url1'],'"></p>';?></div></div>
					
                    <div class="col-lg-6">
					
                        <h2><strong><?php echo $rs['c_name']; ?></strong></h2>
                        <!--<p><strong>商品賣家:</strong></p>-->
                        <p><strong>商品價格:</strong> <?php echo $rs['price']; ?></p>
                        <p><strong>商品尚有存量:</strong><?php echo $rs['quantity']; ?></p>
                        <p><strong>商品描述:</strong><?php echo $rs['describe']; ?></p>
						<?php 
						
						echo '<form action="../cart/cart-insert2.php" method="post">';
                            echo '<input type="hidden" name="thing_id" value="',$rs['c_id'],'">';
							echo '<input type="hidden" name="name" value="',$rs['c_name'],'">';
							echo '<input type="hidden" name="price" value="',$rs['price'],'">';
							echo '<p>數量：<select name="amount" >';
							for ($i=1; $i<=$rs['quantity']; $i++) {
								echo '<option value="', $i, '">', $i, '</option>';
							}
                            echo '</select></p>';
                            echo '<input type="hidden" name="url1" value="',$rs['url1'],'">';
							echo '<p><input type="submit" value="放入購物車"></p>';
							echo '</form>';
							?>
							
                    </div>
                </div>
				<?php
} //while
?>
            </div>
        </section>
        
        
        
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/js/bootstrap.bundle.min.js"></script>
        
        <script src="js/scripts.js"></script>
    </body>
</html>
